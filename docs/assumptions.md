# Model Description / Assumptions Document

## Context and Justification

To set the scene, our simulation is set in **Accra, Ghana**. This is justified because:

- Zipline already operates drone delivery in Ghana, serving over 2,300 health facilities, proving the infrastructure exists.  
- Health workers in Ghana can place an order by text and receive delivery within 30 minutes, serving as a real-world benchmark to beat or match. *(The Borgen Project)*  

---

## Reference Drone Specs: DJI FlyCart 30

| Parameter | Value |
|----------|------|
| Cruise speed | 15 m/s (54 km/h) |
| Max speed | 20 m/s (72 km/h) |
| Flight range (no load) | 28 km |
| Flight range (full load) | 16 km |
| Max flight time (loaded) | 18 minutes |
| Max wind resistance | 12 m/s (43 km/h) |
| Recharge time (single battery) | 26.5 minutes |

The DJI FlyCart 30 flies at a constant **15 m/s in windless conditions**, covering **16 km with a full 30 kg payload** on dual batteries in 18 minutes.

---

## Battery Abstraction and Drain Rates

- Battery is modeled as a **percentage (0–100%)**, not watt-hours.

### Derived Drain Rates

- Full load:  
  `100% ÷ 16 km = 6.25% per km`

- No load:  
  `100% ÷ 28 km = 3.57% per km`

- **Base drain rate (typical package): 5% per km**  
- **Return trip (no package): 3.57% per km**

---

## Delivery Distance Abstraction (Accra Geography)

Assume a warehouse hub in **Tema Industrial Area** delivering across Greater Accra.

| Zone | Distance | % of Orders |
|------|---------|------------|
| Inner city (Osu, Adabraka) | 3 – 6 km | 40% |
| Mid-range (Lapaz, Madina) | 7 – 11 km | 40% |
| Outer (Tema, Ashaiman) | 12 – 16 km | 20% |

- Model: `Uniform(3, 16) km` per delivery  
- Round trip: `2 × distance`  

### Time per Delivery
\[
\text{Time} = \frac{\text{distance}}{54} \times 60 \approx 3–18 \text{ minutes per trip}
\]

---

## Wind Abstraction (Accra Seasonal Data)

Wind speed varies seasonally, affecting battery consumption.

| Season | Months | Avg Wind (km/h) | Battery Multiplier |
|--------|--------|----------------|-------------------|
| Dry / Harmattan | Nov–Feb | 11.6 | ×1.0 |
| Pre-rains | Mar–Apr | 13–15 | ×1.15 |
| Rainy Season | May–Jun | 15–16 | ×1.25 |
| Mid-dry (cool) | Jul–Sep | 15–16 | ×1.25 |
| Post-rains | Oct | 13 | ×1.10 |

- Baseline wind: **11.6 km/h**
- Wind impact is modeled as a **multiplicative increase in battery drain**

---

## Package Arrival Rates

- Zipline capacity: up to **500 flights/day**
- Scaled assumption: **50–80 deliveries/day**

### Hourly Rate
- ~4–7 packages/hour  
- Model: `Poisson(λ = 5/hour)`  
- Average: **1 package every ~12 minutes**

### Heavy Packages
- Probability: `p = 0.30`
- Effect: **battery drain ×1.5**

---

## Battery and Charging Logic

| Parameter | Value | Justification |
|----------|------|--------------|
| Battery threshold | 25% | Safety buffer |
| Recharge time | ~20 minutes | Scaled from 26.5 min |
| Travel to charger | +2 min | Adjacent to warehouse |

- If battery < 25% after delivery → drone enters charging queue  

---

## Model Parameters

| Parameter | Value | Source |
|----------|------|--------|
| DRONE_SPEED | 54 km/h | DJI |
| BATTERY_DRAIN_NORMAL | 5% per km | Derived |
| BATTERY_DRAIN_HEAVY | 7.5% per km | Assumption |
| BATTERY_DRAIN_WIND | 6.25% per km | Seasonal |
| BATTERY_THRESHOLD | 25% | Spec |
| RECHARGE_TIME | 20 min | Derived |
| DELIVERY_DISTANCE | Uniform(3,16) km | Geography |
| ARRIVAL_RATE | Poisson(5/hr) | Zipline scaling |
| HEAVY_PACKAGE_PROB | 0.30 | Assumption |
| SIM_DURATION | 900 min | Operating hours |
| REPLICATIONS | 30 | Statistical validity |
| N_DRONES | Variable | Experimental |
| N_CHARGERS | Variable | Experimental |

---

## Entities and Assumptions

### Packages

- Arrivals follow a **Poisson process**
- Delivery distance: `Uniform(3,16)`
- 30% are heavy (`Bernoulli p=0.30`)
- Heavy packages increase drain by **1.5×**
- Delivery time includes **queue waiting time**

---

### Drones

- One package at a time  
- Start at **100% battery at warehouse**  
- Return to charge at **<25% battery**

#### Noise Modeling
- Process noise: `N(0, 0.5²)` per km  
- Measurement noise: `N(0, 1.5²)`  

#### Environmental Effects
- Wind multiplier based on season (1.0–1.25)

- No failures or maintenance modeled

---

### Charging Stations

- Modeled as **SimPy Resource**
- Capacity = `N_CHARGERS`

#### Charging Time
\[
\text{Time} = \frac{\text{battery deficit}}{100} \times 26.5
\]

- Example: 25% → 100% ≈ 20 minutes  
- Located at warehouse  
- No downtime or failures  

---

## General Model Assumptions

- Simulation duration: **900 minutes (7am–10pm)**  
- SLA target: **95% delivered within 120 minutes**  
- Orders <3 km handled by ground transport  
- 50–80 deliveries/day assumed  

### Statistical Validity
- 30 replications with fixed seeds (1–30)  
- Ensures reliable **95% confidence intervals**

### Routing
- Straight-line distances (≤15% error)  
- Flat terrain assumption  

### System Scope
- Single hub, single fleet  
- No air traffic or collision modeling  
- No weather grounding beyond wind effect  

---

## References

- DJI Technology Co. (2024). *DJI FlyCart 30 Specifications*  
  https://www.dji.com/flycart-30/specs  

- Weather Atlas (2024). *Accra Climate Data*  
  https://www.weather-atlas.com/en/ghana/accra-climate  

- Zipline International (2024). *Ghana Operations*  
  https://flyzipline.com  

- Goebel, K. et al. (2017). *Prognostics: The Science of Making Predictions*  
  doi:10.2514/6.2017-1515  

- Lin, Q. et al. (2020). *COVID-19 Model Study*  
  International Journal of Infectious Diseases  

- Law, A.M. (2015). *Simulation Modeling and Analysis (5th ed.)*  
  McGraw-Hill Education  