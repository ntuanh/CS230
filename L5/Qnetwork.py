import random

a_right = [0, 0.91, 0.83, 0.19, 0]
a_left = [0, 0.49, 0.25, 0.60, 0]
r = [0, 0, 0, 0, 10]

gamma = 0.9
epsilon = 0.2

s = 1
n_loop = 0
n_reach_to_treasure = 0

while True :
    # coin toss : if probability is < 20 % or Q of right = Q of left
    if random.uniform(0, 1) < epsilon or a_right[s] == a_left[s]:
        action_is_right = random.choice([True, False])
    else:
        action_is_right = a_right[s] > a_left[s]

    if action_is_right:
        s_next = s + 1
        a_right[s] = r[s_next] + gamma * max(a_right[s_next], a_left[s_next])
        action_str = "Right"
    else:
        s_next = s - 1 if s > 1 else 1
        a_left[s] = r[s_next] + gamma * max(a_right[s_next], a_left[s_next])
        action_str = "Left"

    print(f"Loop : {n_loop} | State: {s} -> {action_str} -> {s_next}")

    for i in range(1, 4):
        print(f"box {i} [L:{a_left[i]:.2f} R:{a_right[i]:.2f}] |", end=" ")
    print("\n" + "-" * 50)

    if s_next == 4:
        print(f"Reach to Treasure  {n_reach_to_treasure}!.\n")
        n_reach_to_treasure += 1
        if n_reach_to_treasure == 3 :
            break

        s = 1  # Reset
    else:
        s = s_next