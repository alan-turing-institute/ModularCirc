# Models directory
This file contains the description of the model and model parameter classes defined in this module.
The models presented here are made of three types of components:
- *arteries*: RLC components (complex) and R components (simple)
- *chamber*: linear elatic (simple) and mixed (complex, linear active and exponential passive behaviour)
- *valves*: non-ideal diodes (simple), simple Bernoulli (medium), Maynard valves (complex, where the motion the valves is modelled).

## 1. Naghavi et al. model
Relevant files and classes:
- `ModularCirc/Models/NaghaviModel.py`, where the `NaghaviModel` class is defined
- `ModularCirc/Models/NaghaviModelParameters.py`, where the `NaghaviModelParameters` class is defined.

A CV model described in Rapid Estimation of Left Ventricular Contractility with a Physics-Informed Neural Network Inverse Modeling Approach (https://arxiv.org/html/2401.07331v1).
The model is comprised of the following components:
- LA: linear time-varying elastance model (6 parameters)
- MV: non-ideal diode model (1 parameter)
- LV: linear time-varying elastance model (5 parameters)
- AV: non-ideal diode model (1 parameter)
- Aorta: RC Windkessel model (2 parameters)
- Vena cava: RC Windkessel model (2 parameters)

**Total set of parameters sums up to 17.**


[<img src='Figures/NaghavidModel_circut.png'>]()

## 2. Korakianitis and Shi model (V1)
Relevant files and classes:
- `ModularCirc/Models/KorakianitisModel.py`, where the `KorakianitisModel` class is defined
- `ModularCirc/Models/KorakianitisModel_parameters.py`, where the `KorakianitisModel_parameters` class is defined

A simplified CV model described in A concentrated parameter model for the human cardiovascular system
including heart valve dynamics and atrioventricular interaction (https://www.sciencedirect.com/science/article/pii/S1350453305002195?via%3Dihub).
**Here, we simplified the model by eliminating (1) the motion of the annulus fibrosus and (2) the motion of the leaflets, replaced with a simple Bernoulli model.**
This model is comprised of the following components:
- left atrium: linear time-varying elastance model
    - **6 parameters**
- mitral valve: simple Bernoulli model
    - **2 parameters**
- left ventricle: linear time-varying elastance model
    - **5 parameters**
- aortic valve: simple Bernoulli model
    - **2 parameters**
- aortic sinus (RLC 3 component windkessel)
    - **4 parameters**
- arteries (RLC 3 component windkessel) + arteriole (R) + capilary (R)
    - for practical reasons the 3 resistors are summed up into one parameter
    - **4 parameters**
-  systemic venous system (RLC 3 component windkessel)
    - **3 parameters** (assume that venous impedance is zero)
- right atrium: linear time-varying elastance model
    - **6 parameters**
- tricuspid valve: simple Bernoulli model
    - **2 parameters**
- right ventricle: linear time-varying elastance model
    - **5 parameters**
- pulmonary valve: simple Bernoulli model
    - **2 parameters**
- pulmonary artery sinus (RLC 3 component windkessel)
    - **4 parameters**
- pulmonary arteries (RLC 3 component windkessel) + pulmonary arteriole (R) + pulmonary capilary (R)
    - **4 parameters**
- pulmonary venous system (RLC 3 component windessel)
    - **3 parameters** (assume that venous impedance is zero)

**Total set of parameters sums up to 55.**

[<img src=Figures/KorakianitisModel_circuit.png>]()


## 3. Korakianitis and Shi model (V2)
Relevant files and classes:
- `ModularCirc/Models/KorakianitisMaynardModel.py`, where the `KorakianitisMaynardModel` class is defined
- `ModularCirc/Models/KorakianitisMaynardModel_parameters.py`, where the `KorakianitisMaynardModel_parameters` class is defined

This model follow the same strucutre as **Korakianitis and Shi model (V1)**, however the valves are replaced with Maynard type valves which are more similar to the ones used in the original paper.
**This introduces 2 additional parameters per valve (one for the valve closing rate and one for the opening rate).**

Components used:
- arteries: RLC with sinuses being represented as a separate RLC component. (3/4 components)
- valves: **Maynard** valves (4 parameters)
- chambers: **linear models** (5/6 parameters of ventricles/arteries)

**Total set of parameters sums up to 60.**


## 4. Korakianitis and Shi model (V3)
Relevant files and classes:
- `ModularCirc/Models/KorakianitisMixedMaynardModel.py`, where the `KorakianitisMixedMaynardModel` class is defined
- `ModularCirc/Models/KorakianitisMixedMaynardModel_parameters.py`, where the `KorakianitisMixedMaynardModel_parameters` class is defined.

This model is again based on the orinal structure of **Korakianitis and Shi model (V1)**.
Here, the following components were used:
- chambers: **mixed models** (6/7 parameters of ventricles/arteries)
- valves: **Maynard** valves (4 parameters)
- arteries: RLC with sinuses being represented as a separate RLC component. (3/4 parameters)

**Total parameter count: 64**


## 5. Simplified Korakianitis and Shi model (V1)
Relevant files and classes:
- `ModularCirc/Models/MixedHeartMaynard4eWindkessel.py`, where the `MixedHeartMaynard4eWindkessel` class is defined
- `ModularCirc/Models/MixedHeartMaynard4eWindkessel_parameters.py`, where the `MixedHeartMaynard4eWindkessel` class is defined.

This model is again based on the orinal structure of **Korakianitis and Shi model (V1)**.
Here, the following components were used:
- chambers: **mixed models** (6/7 parameters of ventricles/arteries)
- valves: **Maynard** valves (4 parameters)
- arteries: RLC with sinuses represented as a separate R component and capilaries similarly by R. This equivalent to the systemic and pulmonary arterial system being simulated as 2 4-component Windkessels in series with one another.  (7/8 parameters systemic + 7/8 parameters pulmonary)

**Total parameter count: 58 or 56 (if you choose to add the capilary resistance to the resistance of the previous component)**
