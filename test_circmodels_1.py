from ModularCirc.Models.NaghaviModel import NaghaviModel, NaghaviModelParameters
from ModularCirc.Solver import Solver
from ModularCirc.HelperRoutines import activation_function_1, activation_function_2, get_softplus_max
import ModularCirc.StateVariable as sv 
import matplotlib.pyplot as plt
import numpy as np

TEMPLATE_TIME_SETUP_DICT = {
    'name'    :  'generic',
    'ncycles' :  40,    # 30
    'tcycle'  :  800., # 800
    'dt'      :  1.0,
 }

parobj = NaghaviModelParameters()
parobj.set_chamber_comp('lv', tau=80.) # 100.
parobj.set_chamber_comp('la', tau=10.) # 100.
parobj.set_rc_comp('ao', c=0.0025)
parobj.set_activation_function('lv', activation_func=activation_function_1)
parobj.set_activation_function('la', activation_func=activation_function_1)
parobj.set_valve_comp('av', max_func=get_softplus_max(0.2))

model = NaghaviModel(TEMPLATE_TIME_SETUP_DICT, parobj=parobj)
solver = Solver(model=model)
solver.setup()

# print()
# raise Exception
solver.solve()

print(solver.model.all_sv_data)

ind1 = -2*model.time_object.n_c+1
# ind1 = 1
ind2 = -1
# ind2 = model.time_object.n_c


v_lv = solver.model.commponents['lv'].V.values[ind1:ind2]
p_lv = solver.model.commponents['lv'].P_i.values[ind1:ind2]

v_la = solver.model.commponents['la'].V.values[ind1:ind2]
p_la = solver.model.commponents['la'].P_i.values[ind1:ind2]

fig, ax = plt.subplots(ncols=2)
ax[0].plot(v_lv, p_lv)
# ax[0].plot(v_la, p_la)
ax[0].set_title('PV loops')

t = model.time_object._sym_t[ind1:ind2].values
t -= t[0]
q_mv = solver.model.commponents['mv'].Q_i.values[ind1:ind2]
q_av = solver.model.commponents['av'].Q_i.values[ind1:ind2]
ax[1].plot(t, q_mv)
ax[1].plot(t, q_av)
ax[1].set_title('Valve flow')
p_ao = solver.model.commponents['ao'].P_i.values[ind1:ind2]
q_ao = solver.model.commponents['ao'].Q_o.values[ind1:ind2]



fig2, ax2 = plt.subplots(nrows=2, ncols=2)

ax2[0][0].plot(t, p_lv, label='lv')
ax2[0][0].plot(t, p_ao, label='ao')
ax2[0][0].plot(t, p_la, label='la')
ax2[0][0].set_title('Pressure')
ax2[0][0].legend()

ax2[0][1].plot(t, q_mv, label='mv')
ax2[0][1].plot(t, q_ao, label='ao')
ax2[0][1].plot(t, q_av, label='av')
ax2[0][1].set_title('Q')
ax2[0][1].legend()

af_lv = np.vectorize(model.commponents['lv']._af) 
af_la = np.vectorize(model.commponents['la']._af) 
tc = model.time_object._cycle_t[ind1:ind2]
ax2[1][0].plot(t, af_la(tc), label='la')
ax2[1][0].plot(t, af_lv(tc), label='lv')
ax2[1][0].set_title('activation')
ax2[1][0].legend()

ax2[1][1].plot(t, v_lv, label='lv')
ax2[1][1].plot(t, v_la, label='la')
ax2[1][1].set_title('vol')
ax2[1][1].legend()

plt.show()