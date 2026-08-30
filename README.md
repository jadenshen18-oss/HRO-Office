# 🖥️ HRO-Zone1
Home Resource Orchestrator is an Arduino UNO Q-based smart home monitoring system designed to reduce unnecessary household energy and resource consumption while detecting potentially dangerous situations such as unattended faucets, running toilets, and kitchen appliances left operating in empty rooms.

#### What's the difference between this project and traditional smart home systems?
HRO uses occupancy, enviornemntal, audio, and resource information to understand the context behind what is happening in the home.

## Current Prototype:
The current prototype demonstrates the basic **person-detection** and automatic **energy-saving** system.

A camera is used to detect whether a person is present. Based on this detection, the system automatically controls a light and a fan:

Person detected → Light and fan turn **ON**
No person detected → Light and fan turn **OFF**

This demonstrates how the system can reduce unnecessary energy use by making sure devices are only running when someone is actually present to use them.


#### A Streamlit dashboard: 
- a project photo
- an LED toggle control
- progress status bar
- semi-circle pie charts
- person detection model result
- line chart sourced from mydata.csv.

## Future Plan
In the future, we plan to expand the system beyond vision by giving it additional ways to sense what is happening throughout a home.

For example, sound detection could listen for patterns such as continuously running water. If a faucet or toilet appears to have been running for an unusually long time, the system could warn the homeowner or automatically shut off the water.

Other environmental sensors could detect smoke, unusual gases, temperature changes, humidity, or other warning signs. These could allow the system to identify possible hazards in addition to resource waste.

By combining multiple types of sensing, the long-term goal is to create a system that can understand more of what is happening around a home and respond appropriately:

Sight → Detect people and occupancy
Sound → Detect running water or unusual appliance sounds
Environmental sensors → Detect smoke, gases, temperature, and humidity
Physical controls → Control lights, fans, appliances, or water valves

The current light and fan prototype is the first demonstration of this larger idea: a home system that can sense its environment, recognize unnecessary resource use or potential hazards, and automatically take action.
