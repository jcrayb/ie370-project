import numpy as np
import pandas as pd


## project constants
max_tickets = 10
prob_of_buying = np.array([1, .4])
prob_of_happening = np.array([1/4, 1/2, 1/4])
price = np.array([100, 200])

possible_actions = [(max(j-i, 0), min(i, j)) \
                        for j in range(0, 3) \
                        for i in range(0, 3) if i <= j]

actions_dict = {i: action for i, action in enumerate(possible_actions)}

v0 = []
for action in possible_actions:
    customer_array = np.array([[min(ncostumers, action[0]), min(action[1], max(0, ncostumers-action[0]))] \
                                for ncostumers in range(0, 3)])
    v0 += [customer_array*prob_of_buying@price@prob_of_happening]

P_0_0 = np.eye(11)

pstart_r = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
P_1_0 = np.array([pstart_r] +[np.roll(pstart_r, i)*3/4 + np.roll(pstart_r, i+1)*1/4 for i in range(0, 10)])

pstart_p = np.array([1/2*.4+.4*1/4, 1/4+.6*1/2+.6*1/4, 0, 0, 0, 0, 0, 0, 0, 0, 0])
P_0_1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]+[np.roll(pstart_p, i) for i in range(0, 10)])

P_2_0 = np.array([pstart_r*.75]*2+[np.roll(pstart_r, i)*1/4 + np.roll(pstart_r, i+1)*1/2 for i in range(0, 9)])+ np.eye(11)*1/4

pstart_rp = np.array([.1, .65, 0, 0, 0, 0, 0, 0, 0, 0, 0])
P_1_1 = np.array([pstart_r*.75]*2+[np.roll(pstart_rp, i) for i in range(0, 9)]) + np.eye(11)/4

pstart_pp = np.array([1/4*.4**2, 1/2*.4+1/2*.6*.4, 1/4+1/2*.6+1/4*.6**2, 0, 0, 0, 0, 0, 0, 0, 0])
P_0_2 = np.array([pstart_r]+[np.array([1-(1/4+1/2*.6+1/4*.6**2), 1/4+1/2*.6+1/4*.6**2, 0, 0, 0, 0, 0, 0, 0, 0, 0])]+[np.roll(pstart_pp, i) for i in range(0, 9)])

Pk = [P_0_0, P_1_0, P_0_1, P_2_0, P_1_1, P_0_2]

optimal_policy = {}
optimal_value = {}

v_star = np.zeros(11)

for day in range(15):
    v = v_star
    a, v_star = [], []
    for i in range(max_tickets+1):
        v_temp = [v0[j] + Pk[j][i]@v if sum(actions_dict[j]) <=  i else 0 for j in range(6)]
        v_star += [max(v_temp)]
        
        a_star = np.argmax(v_temp)
        a += [possible_actions[a_star]]
    optimal_value[day] = v_star
    optimal_policy[day] = a

action_df = pd.DataFrame(optimal_policy).T
action_df['Days left'] = [i for i in range(0, 15)]
action_df.set_index('Days left')

print(action_df)

value_df = pd.DataFrame(optimal_value).T
value_df['Days left'] = [i for i in range(0, 15)]
value_df.set_index('Days left')

print(value_df)