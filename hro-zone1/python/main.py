import asyncio
import threading
import time
from collections import deque
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit.components.v1 as components
from kasa import Discover

from arduino.app_utils import App
from arduino.app_utils import Logger
from arduino.app_bricks.streamlit_ui import st
from arduino.app_bricks.video_imageclassification import VideoImageClassification

app_logger = Logger("HRO-Zone1")

# Sets the browser tab title, which is separate from the st.title() heading and
# is otherwise left at whatever the brick configured. It has to be the first
# Streamlit call, and throws if the brick already set it -- hence the guard.
try:
    st.set_page_config(page_title="HRO-Zone1")
except Exception:
    pass

# Categorical slots 1-4, dark-surface steps: identity colours assigned in fixed
# order, never cycled.
SERIES_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500"]

# Height of the Office Ceiling Fan power-usage chart.
PANEL_HEIGHT = 200

# CAMERA_BASE_HEIGHT is where the /embed page's video actually renders at its
# native fixed size (independent of PANEL_HEIGHT); CAMERA_HEIGHT_SCALE then
# visually magnifies it via CSS transform in camera_stream_embed() -- see
# that function's docstring for why a taller box alone doesn't work.
CAMERA_BASE_HEIGHT = 220
CAMERA_HEIGHT_SCALE = 4 / 3  # previous 2x reduced by a third, per feedback

# The detection callback fires on the brick's own background thread, not
# Streamlit's script thread, so the shared state it writes to needs a lock.
# A bounded deque keeps only the most recent detections, mirroring the
# "recent detections" list from the original objectdetectcheck app.
DETECTION_HISTORY_LEN = 8
_detection_lock = threading.Lock()
_recent_detections = deque(maxlen=DETECTION_HISTORY_LEN)

# Unlike detection, classification always produces a label for every frame --
# "person" or "not-person" -- so the callback fires continuously whether or
# not anyone's in view. person_detected can just be set directly from each
# frame's result; no staleness watchdog is needed to notice an absence that
# never gets reported.
person_detected = False

# The brick calls this callback for every frame it processes, which can be
# faster than the board can comfortably keep up with. Skipping straight to
# `return` below the throttle check keeps that cost capped at this rate.
MIN_FRAME_INTERVAL_SEC = 1.0
_last_processed_at = 0.0

# Automation timing: turning ON reacts instantly (no reason to make someone
# wait for the fan/lights once they walk in), but turning OFF waits for a
# continuous absence, so a person briefly stepping out of frame doesn't
# flick the plugs off and back on. _last_person_seen_at is None until the
# first-ever detection, so the away-timer can't fire from a false "away"
# baseline set at app startup before anyone has been seen at all.
AUTOMATION_OFF_DELAY_SEC = 15.0
_last_person_seen_at = None
_automation_off_fired = False
# Set (fast, no I/O) by the classification callback when a person just
# arrived; consumed by _automation_watchdog_loop, which does the actual
# (potentially slow) plug communication. Keeping any network I/O out of
# _on_all_classifications matters: that callback runs on the video
# classification brick's own background thread, and a Tapo device that's
# slow to respond (as seen in testing -- repeated 403s/decrypt errors) would
# block that thread and stall new frames from being processed at all, which
# is what broke the recent-detections history.
_pending_turn_on = False


def _on_all_classifications(classifications: dict):
    """Runs on the classification brick's background thread for every frame.

    `classifications` is {label: confidence} -- a flat dict, unlike the
    face-detection brick's {label: [{confidence, bbox, ...}, ...]}, since a
    classifier scores the whole frame rather than localising boxes. The
    person-classification model reports "person" and/or "not-person".

    Unlike VideoObjectDetection, this brick's constructor has no
    camera_preview option (confirmed: passing it raises TypeError) and this
    callback receives no frame argument, so there's no way to show the live
    camera image through this API -- only the classification result.

    Only touches shared in-memory state here -- no network calls. See
    _pending_turn_on and _automation_watchdog_loop for why."""
    global person_detected, _last_processed_at, _last_person_seen_at, _automation_off_fired, _pending_turn_on
    try:
        now = time.monotonic()
        if now - _last_processed_at < MIN_FRAME_INTERVAL_SEC:
            return
        _last_processed_at = now

        with _detection_lock:
            previously_detected = person_detected
            person_score = classifications.get("person", 0.0)
            not_person_score = classifications.get("not-person", 0.0)
            person_detected = "person" in classifications and person_score >= not_person_score
            for label, confidence_value in classifications.items():
                _recent_detections.appendleft(
                    {"object": label, "confidence": confidence_value}
                )
            just_arrived = person_detected and not previously_detected
            if person_detected:
                _last_person_seen_at = now
            if just_arrived:
                _pending_turn_on = True
                _automation_off_fired = False
    except Exception as e:
        # Runs on the brick's own background thread -- an uncaught exception
        # here would otherwise be invisible (no Streamlit error box to catch
        # it) and could silently stop future frames from being processed.
        app_logger.error(f"Classification callback failed: {e!r}")


def _automation_watchdog_loop():
    """Polls independently of the classification callback (which only runs
    when a new frame arrives, and must stay non-blocking -- see
    _pending_turn_on) so the 5s away-timer still fires even if frames stop
    coming in entirely, and so the actual (potentially slow) plug
    communication never blocks video processing. Turns off at most once per
    absence -- _automation_off_fired blocks repeat calls until the person is
    next seen (which resets it), so a manual ON in the meantime is left
    alone rather than being fought every half second."""
    global _automation_off_fired, _pending_turn_on
    while True:
        time.sleep(0.5)
        try:
            with _detection_lock:
                turn_on_now = _pending_turn_on
                _pending_turn_on = False
                detected = person_detected
                last_seen = _last_person_seen_at
                already_fired = _automation_off_fired

            if turn_on_now:
                set_fan_power(True)
                set_kitchen_power(True)
                continue

            if detected or already_fired or last_seen is None:
                continue
            if time.monotonic() - last_seen >= AUTOMATION_OFF_DELAY_SEC:
                with _detection_lock:
                    _automation_off_fired = True
                set_fan_power(False)
                set_kitchen_power(False)
        except Exception as e:
            app_logger.error(f"Automation watchdog failed: {e!r}")


@st.cache_resource
def init_video_detector():
    """Instantiated once per app process. Streamlit re-runs this whole script on
    every interaction; without caching, each rerun would reopen the camera and
    re-register the callback. Returns None on failure (e.g. no camera
    connected) instead of raising, so the rest of the page can still render
    with a friendly message -- st.cache_resource does not cache exceptions,
    so a raise here would re-attempt (and re-fail) on every single rerun."""
    try:
        detector = VideoImageClassification(confidence=0.5, debounce_sec=0.0)
        detector.on_detect_all(_on_all_classifications)
        return detector
    except Exception as e:
        app_logger.error(f"Could not initialize the video detector: {e!r}")
        return None


# Defined once at module level rather than inside the script body -- a fragment
# redefined fresh on every full rerun (e.g. from a nested `def` inside a
# `with side:` block) gets a new identity each time, orphaning the previous
# instance's run_every schedule (confirmed via the "fragment with id ... does
# not exist anymore" warning in apps_logs). A stable top-level function keeps
# the same identity, so its 1s auto-refresh keeps working across reruns.
@st.fragment(run_every="1s")
def live_camera_view():
    with _detection_lock:
        rows = list(_recent_detections)
        currently_detected = person_detected

    st.write("Person detected: " + ("Yes" if currently_detected else "No"))

    panel_title("Recent Detections")
    if rows:
        table = pd.DataFrame(rows)
        table["confidence"] = table["confidence"].map(
            lambda c: f"{c:.0%}" if c is not None else "—"
        )
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.write("No objects detected yet")


def camera_stream_embed(base_height: int = CAMERA_BASE_HEIGHT, height_scale: float = CAMERA_HEIGHT_SCALE):
    """Live camera feed via the model-runner's own :4912/embed endpoint --
    the same one the "Person Classifier on Camera" example points its iframe
    at (see that example's app.js). VideoImageClassification's Python
    callback has no frame data (confirmed: no camera_preview option), so
    this is the only way to show live video for this brick.

    This runs as real JavaScript in the visitor's browser via
    components.html() -- the hostname has to resolve from there, not from
    the board's internal Docker network, which is why a server-side Python
    request to this URL can't substitute for it (it lands on the wrong
    network and, separately, returns the HTML viewer page itself rather
    than raw image bytes -- it's meant to be embedded, not parsed).

    components.html() renders its content inside an <iframe srcdoc="...">,
    which has an opaque origin -- window.location.hostname is permanently
    empty in there, confirmed by direct testing, so it can't be used to find
    the real host. document.referrer on a srcdoc iframe is set to the parent
    page's real URL instead, so that's the source of truth here.

    The /embed page's video renders at a fixed intrinsic size regardless of
    the iframe's own width/height (confirmed by direct testing) -- simply
    reserving a taller box only added blank space below the video, which was
    also what pushed "Person detected" and the recent-detections table
    further down the page. `transform: scaleY()` instead visually magnifies
    the already-rendered pixels, so the reserved wrapper height matches the
    scaled-up size exactly, with no leftover blank space. Only Y is scaled
    (not X) since the width was already sized correctly by the column.

    Built with placeholder tokens + str.replace() rather than an f-string or
    .format(): the JS below has its own { } blocks (and a ${...} template
    literal) that an f-string/format string would otherwise misparse or
    require escaping throughout."""
    total_height = int(base_height * height_scale)
    inner_height_pct = 100 / height_scale
    html = """
    <div style="width:100%;height:__TOTAL_HEIGHT__px;position:relative;overflow:hidden;">
      <div id="videoPlaceholder"
           style="color:#898781;font-family:system-ui,sans-serif;text-align:center;padding-top:40%;">
        Connecting to camera stream…
      </div>
      <iframe id="dynamicIframe" title="Video stream"
              style="display:none;position:absolute;top:0;left:0;border:none;
                     width:100%;height:__INNER_PCT__%;
                     transform:scaleY(__SCALE__);transform-origin:top left;"></iframe>
    </div>
    <script>
      const iframe = document.getElementById('dynamicIframe');
      const placeholder = document.getElementById('videoPlaceholder');

      let hostname = window.location.hostname;
      try {
        if (document.referrer) {
          hostname = new URL(document.referrer).hostname;
        }
      } catch (e) {
        // fall through to window.location.hostname
      }
      const streamUrl = `http://${hostname}:4912/embed`;

      let intervalId;
      iframe.onload = () => {
        clearInterval(intervalId);
        placeholder.style.display = 'none';
        iframe.style.display = 'block';
      };
      intervalId = setInterval(() => { iframe.src = streamUrl; }, 1000);
    </script>
    """
    html = (
        html.replace("__TOTAL_HEIGHT__", str(total_height))
        .replace("__INNER_PCT__", f"{inner_height_pct:.4f}")
        .replace("__SCALE__", str(height_scale))
    )
    components.html(html, height=total_height)


def button_css(key: str, background: str, text: str) -> str:
    """CSS for one button. A button's key becomes a ".st-key-<key>" class on its
    wrapper, so each button can be targeted individually. Streamlit ships its own
    :hover/:focus/:active colours, so every state is pinned to the same colour --
    otherwise the button keeps Streamlit's hover colour while the cursor is on it
    and only flips to ours once the cursor leaves."""
    states = ":hover", ":focus", ":focus-visible", ":active"
    selectors = ", ".join(f".st-key-{key} button{s}" for s in ("",) + states)
    return f"""
    {selectors} {{
        background-color: {background} !important;
        color: {text} !important;
        border-color: {background} !important;
    }}"""


PANEL_TITLE_COLOR = "#3987e5"  # categorical slot 1 blue, matches the chart/gauge accent elsewhere


def panel_title(text: str):
    """A subheader in PANEL_TITLE_COLOR instead of the page's default white
    h3 -- st.subheader() has no per-call colour option, so this renders the
    same h3 markup directly with an inline style override."""
    st.markdown(f'<h3 style="color: {PANEL_TITLE_COLOR};">{text}</h3>', unsafe_allow_html=True)


def render_power_chart(samples, reference_power):
    """The line chart shared by both plugs' panels, with a "Live Power Usage"
    annotation in the top-left corner. st.line_chart has no annotation/text
    option, so this builds the equivalent chart directly in Altair (which
    st.line_chart itself uses internally) and layers a text mark on top.
    x/y for the annotation are pinned to fixed pixel offsets via alt.value()
    rather than data coordinates, so the label sits in the same corner
    regardless of the data's actual time/power range.

    reference_power fixes the y-axis domain to [0, 2 * reference_power] --
    the first power reading this session -- instead of Altair's default
    auto-scaling tightly around the data, which left the line hugging the
    top of the chart. With this fixed domain, a reading at/near that first
    measurement sits at the middle of the chart instead."""
    chart_data = pd.DataFrame(samples, columns=["time", "Power (W)"])
    y_scale = alt.Scale(domain=[0, reference_power * 2]) if reference_power else alt.Undefined
    line = (
        alt.Chart(chart_data)
        .mark_line(color=SERIES_COLORS[0])
        .encode(x=alt.X("time:T", title=None), y=alt.Y("Power (W):Q", scale=y_scale))
    )
    annotation = (
        alt.Chart(pd.DataFrame({"text": ["Live Power Usage"]}))
        .mark_text(align="left", baseline="top", dx=6, dy=6, fontSize=13, fontWeight="bold", color="#898781")
        .encode(x=alt.value(0), y=alt.value(0), text="text:N")
    )
    st.altair_chart((line + annotation).properties(height=PANEL_HEIGHT), width="stretch")


# --- Tapo P115 smart plugs (Office Ceiling Fan, Kitchen Room Lights) ----
FAN_NAME = "Office Ceiling Fan"
FAN_HOST = "192.168.1.57"
KITCHEN_NAME = "Kitchen Room Lights"
KITCHEN_HOST = "192.168.1.73"
TAPO_EMAIL = "cvsdhome@yahoo.com"
TAPO_PASSWORD = "tplink123"
FAN_POLL_INTERVAL_SEC = 10  # was 3s; each poll does a full fresh reconnect/handshake, and
# reconnecting that often turned out to trigger 403s from the plugs themselves (repeated
# re-authentication at this rate looks like it degrades the device's own session state).
FAN_HISTORY_LEN = 60  # 60 samples * 10s = last 10 minutes

# Written by each background poll thread (and by its button's on_click, for
# an immediate reflect instead of waiting for the next poll tick); read by
# the fragments below. Mirrors the _detection_lock / _recent_detections
# pattern above -- a plain module variable would race with Streamlit's
# script thread. One lock/state set per plug -- they poll independently.
_fan_lock = threading.Lock()
_fan_power_samples = deque(maxlen=FAN_HISTORY_LEN)  # [(epoch seconds, watts), ...]
_fan_is_on = False
_fan_last_error = None
_fan_reference_power = None  # set once, from the first reading -- fixes the chart's y-axis domain

_kitchen_lock = threading.Lock()
_kitchen_power_samples = deque(maxlen=FAN_HISTORY_LEN)
_kitchen_is_on = False
_kitchen_last_error = None
_kitchen_reference_power = None


async def _connect_tapo_device(host: str):
    device = await Discover.discover_single(
        host, username=TAPO_EMAIL, password=TAPO_PASSWORD
    )
    await device.update()
    return device


async def _with_tapo_device(host: str, action=None):
    """Connect, optionally run `action` on the device, read is_on/power, then
    always disconnect -- leaving the connection open would leak an aiohttp
    session on every 3s poll. Shared by both plugs; only the host differs."""
    device = await _connect_tapo_device(host)
    try:
        if action:
            await action(device)
            await device.update()
        power = None
        try:
            energy = device.modules.get("Energy")
            if energy:
                value = getattr(energy, "current_consumption", None)
                if callable(value):
                    value = value()
                power = value
        except Exception:
            power = None
        return bool(device.is_on), power
    finally:
        await device.disconnect()


def _fan_poll_loop():
    # One event loop for this thread's whole lifetime, not a fresh one every
    # 3s via asyncio.run() -- Streamlit's own server runs on its own uvloop
    # on the main thread, and constantly creating/tearing down loops in
    # background threads alongside it risked contention (a
    # "RuntimeError: Event loop is closed" surfaced on the main thread while
    # this was still asyncio.run()-per-iteration).
    global _fan_is_on, _fan_last_error, _fan_reference_power
    loop = asyncio.new_event_loop()
    while True:
        try:
            is_on, power = loop.run_until_complete(_with_tapo_device(FAN_HOST))
            with _fan_lock:
                _fan_is_on = is_on
                if power is not None:
                    _fan_power_samples.append((time.time(), power))
                    if _fan_reference_power is None:
                        _fan_reference_power = power
                _fan_last_error = None
        except Exception as e:
            with _fan_lock:
                _fan_last_error = str(e)
        time.sleep(FAN_POLL_INTERVAL_SEC)


def set_fan_power(target_on: bool):
    """Drive the fan to a specific state (used by the toggle button and by
    the person-detection automation below), rather than just flipping
    whatever it currently is."""
    global _fan_is_on, _fan_last_error, _fan_reference_power
    action = (lambda device: device.turn_on()) if target_on else (lambda device: device.turn_off())
    try:
        is_on, power = asyncio.run(_with_tapo_device(FAN_HOST, action))
        with _fan_lock:
            _fan_is_on = is_on
            if power is not None:
                _fan_power_samples.append((time.time(), power))
                if _fan_reference_power is None:
                    _fan_reference_power = power
            _fan_last_error = None
    except Exception as e:
        with _fan_lock:
            _fan_last_error = str(e)


def toggle_fan():
    with _fan_lock:
        turning_on = not _fan_is_on
    set_fan_power(turning_on)


def _kitchen_poll_loop():
    # See _fan_poll_loop's comment: one persistent loop, not asyncio.run()
    # every iteration.
    global _kitchen_is_on, _kitchen_last_error, _kitchen_reference_power
    loop = asyncio.new_event_loop()
    while True:
        try:
            is_on, power = loop.run_until_complete(_with_tapo_device(KITCHEN_HOST))
            with _kitchen_lock:
                _kitchen_is_on = is_on
                if power is not None:
                    _kitchen_power_samples.append((time.time(), power))
                    if _kitchen_reference_power is None:
                        _kitchen_reference_power = power
                _kitchen_last_error = None
        except Exception as e:
            with _kitchen_lock:
                _kitchen_last_error = str(e)
        time.sleep(FAN_POLL_INTERVAL_SEC)


def set_kitchen_power(target_on: bool):
    """See set_fan_power's docstring -- same reasoning, second plug."""
    global _kitchen_is_on, _kitchen_last_error, _kitchen_reference_power
    action = (lambda device: device.turn_on()) if target_on else (lambda device: device.turn_off())
    try:
        is_on, power = asyncio.run(_with_tapo_device(KITCHEN_HOST, action))
        with _kitchen_lock:
            _kitchen_is_on = is_on
            if power is not None:
                _kitchen_power_samples.append((time.time(), power))
                if _kitchen_reference_power is None:
                    _kitchen_reference_power = power
            _kitchen_last_error = None
    except Exception as e:
        with _kitchen_lock:
            _kitchen_last_error = str(e)


def toggle_kitchen():
    with _kitchen_lock:
        turning_on = not _kitchen_is_on
    set_kitchen_power(turning_on)


@st.cache_resource
def start_background_threads():
    """Runs exactly once per app process, like init_video_detector() above --
    without st.cache_resource, this call (being top-level script code) would
    re-run on *every* full script rerun (e.g. moving the confidence slider),
    spawning a fresh set of duplicate poll/watchdog threads each time. Those
    would all hit the same two physical plugs concurrently, which lines up
    with the intermittent 403/decrypt errors and unexpected power-state
    changes seen in testing -- multiple simultaneous sessions to the same
    Tapo device stepping on each other."""
    threading.Thread(target=_fan_poll_loop, daemon=True).start()
    threading.Thread(target=_kitchen_poll_loop, daemon=True).start()
    threading.Thread(target=_automation_watchdog_loop, daemon=True).start()
    return True


start_background_threads()


# Split from the chart below so the button can be absolutely positioned onto
# the title's row (see .st-key-fan_title_row / .st-key-toggle_fan_btn in the
# page-level CSS) while the chart spans the full column underneath. Same
# reasoning as live_camera_view for being a stable top-level function.
@st.fragment(run_every=f"{FAN_POLL_INTERVAL_SEC}s")
def office_fan_toggle():
    with _fan_lock:
        is_on = _fan_is_on

    # Idle is dark grey with light text; on fills with colour and flips its
    # label to dark so it stays readable -- same convention the old LED
    # toggle used.
    fan_color, fan_text = ("#199e70", "#ffffff") if is_on else ("#333333", "#ffffff")
    st.markdown(
        f"""
        <style>
        .st-key-toggle_fan_btn button {{ min-width: 160px !important; }}
        {button_css("toggle_fan_btn", fan_color, fan_text)}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.button(
        "ON" if is_on else "OFF",
        key="toggle_fan_btn",
        on_click=toggle_fan,
    )


@st.fragment(run_every=f"{FAN_POLL_INTERVAL_SEC}s")
def office_fan_chart():
    with _fan_lock:
        samples = list(_fan_power_samples)
        error = _fan_last_error
        reference_power = _fan_reference_power

    if error:
        st.caption(f"Power reading unavailable: {error}")

    if samples:
        render_power_chart(samples, reference_power)
    else:
        st.write("Waiting for first power reading…")


# Same structure as the office_fan_* pair above, for the second plug.
@st.fragment(run_every=f"{FAN_POLL_INTERVAL_SEC}s")
def kitchen_lights_toggle():
    with _kitchen_lock:
        is_on = _kitchen_is_on

    fan_color, fan_text = ("#199e70", "#ffffff") if is_on else ("#333333", "#ffffff")
    st.markdown(
        f"""
        <style>
        .st-key-toggle_kitchen_btn button {{ min-width: 160px !important; }}
        {button_css("toggle_kitchen_btn", fan_color, fan_text)}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.button(
        "ON" if is_on else "OFF",
        key="toggle_kitchen_btn",
        on_click=toggle_kitchen,
    )


@st.fragment(run_every=f"{FAN_POLL_INTERVAL_SEC}s")
def kitchen_lights_chart():
    with _kitchen_lock:
        samples = list(_kitchen_power_samples)
        error = _kitchen_last_error
        reference_power = _kitchen_reference_power

    if error:
        st.caption(f"Power reading unavailable: {error}")

    if samples:
        render_power_chart(samples, reference_power)
    else:
        st.write("Waiting for first power reading…")


video_detector = init_video_detector()

st.title("HRO-Zone1")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp p {
        color: #ffffff;
    }
    /* st.title() renders as the page's only h1; the h2/h3 subheaders inside the
       columns are untouched, so only the top title gets a full-width gradient
       banner -- reusing the blue/teal already used for the chart series 1 and
       3, so it reads as part of the same palette rather than a random accent
       colour -- plus the extra bottom margin that pushes the columns below it
       down a few lines. */
    .stApp h1 {
        text-align: center;
        margin: 0 0 2.5rem 0;
        padding: 1rem 0;
        font-weight: 800;
        letter-spacing: 0.12em;
        color: #ffffff;
        background: linear-gradient(90deg, #3987e5, #199e70);
        border-radius: 12px;
        text-shadow: 0 0 16px rgba(0, 0, 0, 0.35);
    }
    div[data-testid="stButton"] button {
        border-radius: 999px;
        border: none;
        padding: 0.5rem 1.5rem;
    }
    /* The streamlit_ui brick's own set_page_config() call runs before this
       script (confirmed: our own call throws "already configured" and is
       swallowed by the try/except above), so its default "centered" layout
       can't be overridden through the API -- only a max-width override on
       Streamlit's content wrapper actually widens the page on a large
       monitor. Every selector here targets that same wrapper under a
       different Streamlit version's testid/class naming, since which one
       is present isn't known ahead of time. */
    div[data-testid="stAppViewBlockContainer"],
    div[data-testid="stMainBlockContainer"],
    section[data-testid="stMain"],
    .main .block-container,
    .block-container {
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    /* Streamlit puts a ~1rem gap between every stacked element by default --
       most visible under the camera_stream_embed() iframe, which otherwise
       leaves the "Person detected" text and table floating well below it. */
    div[data-testid="stVerticalBlock"] {
        gap: 0.1rem;
    }
    /* A plain st.write("") renders near-zero height under the tightened gap
       above (no text content to give it line-height), so column-title
       spacing uses this explicit spacer instead. */
    .title-spacer {
        height: 2.5rem;
    }
    /* Gap between the Office Ceiling Fan and Kitchen Room Lights panels. */
    .panel-gap {
        height: 0.5in;
    }
    /* Puts each ON/OFF button on the title's own row, pinned to its right
       edge. Streamlit always stacks the subheader and the button as
       separate full-width blocks; flexing their shared parent didn't
       change that (unreliable to target across Streamlit versions), and
       floating the button up with a negative margin required guessing its
       exact pixel height and overshot into the text. `position: absolute`
       is deterministic instead: the button is pinned to the top-right
       corner of the *container* wrapping both it and the title, regardless
       of the title's actual rendered height. `key=` on st.container() is
       what puts the ".st-key-..." class on that wrapper. padding-right
       reserves room so long title text can't run under the button. */
    .st-key-fan_title_row,
    .st-key-kitchen_title_row {
        position: relative;
        padding-right: 180px;
        min-height: 2.6rem;
    }
    .st-key-toggle_fan_btn,
    .st-key-toggle_kitchen_btn {
        position: absolute;
        top: 0;
        right: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Two top-level columns: the human-detection panel and the Office Ceiling Fan
# panel. Both start at the top edge so their headings line up. Panel is twice
# as wide as side so the power chart it holds gets twice the room.
side, panel = st.columns([1, 2], gap="large", vertical_alignment="top")

with side:
    panel_title("Human Detection")
    st.markdown('<div class="title-spacer"></div>', unsafe_allow_html=True)

    try:
        if video_detector is None:
            st.error("Camera/model unavailable -- check that a USB camera is connected and restart the app.")
        else:
            confidence = st.slider(
                "Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                key="confidence_slider",
            )
            try:
                video_detector.override_threshold(confidence)
            except Exception as exc:
                # "Connection refused" here means the model-runner's socket isn't up
                # yet -- expected for the first few seconds after this app starts,
                # and it self-resolves once VideoImageClassification's own background
                # thread finishes connecting (see the "WebSocket connection
                # established" line in apps_logs). Only surface genuinely unexpected
                # failures, not this transient startup race.
                if "refused" not in str(exc).lower():
                    st.caption(f"Threshold override unavailable: {exc}")
                else:
                    print(f"[person-detection] threshold override retry pending: {exc!r}")

            # Rendered once per full script run, not inside the fragment -- the
            # iframe manages its own live connection internally, so re-mounting it
            # every 1s (the fragment's rate) would just make it keep reloading.
            camera_stream_embed()

            # Defined at module level (see above) so its run_every schedule survives
            # full-script reruns instead of being orphaned by each one.
            live_camera_view()
    except Exception as e:
        # Isolates this panel from the rest of the page -- an uncaught
        # exception here would otherwise stop the whole script (including
        # the Office Ceiling Fan / Kitchen Room Lights panel below) from
        # rendering at all.
        app_logger.error(f"Human Detection panel failed to render: {e!r}")
        st.error(f"Human Detection panel error: {e}")

with panel:
    try:
        with st.container(key="fan_title_row"):
            panel_title(FAN_NAME)
            office_fan_toggle()
        st.markdown('<div class="title-spacer"></div>', unsafe_allow_html=True)
        office_fan_chart()
    except Exception as e:
        # Each plug's section is isolated from the other -- a problem with
        # one shouldn't stop the second plug's panel from rendering.
        app_logger.error(f"{FAN_NAME} panel failed to render: {e!r}")
        st.error(f"{FAN_NAME} panel error: {e}")

    st.markdown('<div class="panel-gap"></div>', unsafe_allow_html=True)

    try:
        with st.container(key="kitchen_title_row"):
            panel_title(KITCHEN_NAME)
            kitchen_lights_toggle()
        st.markdown('<div class="title-spacer"></div>', unsafe_allow_html=True)
        kitchen_lights_chart()
    except Exception as e:
        app_logger.error(f"{KITCHEN_NAME} panel failed to render: {e!r}")
        st.error(f"{KITCHEN_NAME} panel error: {e}")

App.run()
