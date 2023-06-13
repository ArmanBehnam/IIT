import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Define the function f_A
def f_A(B, C, D, U_A1, U_A2, U_AB, U_AC):
    return f_A_helper(B, U_A1, U_A2, U_AB) and f_A_helper(C, U_A1, U_A2, U_AC) and (not D)

# Helper function for f_A
def f_A_helper(X, U_A1, U_A2, U_Ax):
    return ((not X) ^ U_A1 or U_Ax) ^ U_A2

# Define the functions f_B, f_C, and f_D
def f_B(U_B, U_AB):
    return (not U_B) and (not U_AB)

def f_C(U_C, U_AC):
    return (not U_C) and (not U_AC)

def f_D(U_C):
    return not U_C

def generate_sample(intervention):
    U_A1 = np.random.randint(2)
    U_A2 = np.random.randint(2)
    U_AB = np.random.randint(2)
    U_AC = intervention or np.random.randint(2)
    U_B = np.random.randint(2)
    U_C = np.random.randint(2)
    U_D = np.random.randint(2)

    A = f_A(U_B, U_C, U_D, U_A1, U_A2, U_AB, U_AC)
    B = f_B(U_B, U_AB)
    C = f_C(U_C, U_AC)
    D = f_D(U_D)
    A_given_C_eq_1 = f_A(U_B, True, U_D, U_A1, U_A2, U_AB, U_AC)

    return {'A': A, 'B': B, 'C': C, 'D': D, 'U_A1': U_A1, 'U_A2': U_A2,
            'U_AB': U_AB, 'U_AC': U_AC, 'U_B': U_B, 'U_C': U_C, 'U_D': U_D, 'A_given_C_eq_1': A_given_C_eq_1}

num_samples = 10000
results = []

for i in range(500):
    counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'A_given_C_eq_1': 0}
    counts_AC = 0

    for j in range(num_samples):
        sample = generate_sample(intervention=np.random.randint(2))  # intervene on C by setting it to 1
        if sample['A']:
            counts['A'] += 1
        if sample['B']:
            counts['B'] += 1
        if sample['C']:
            counts['C'] += 1
            if sample['A']:
                counts_AC += 1
        if sample['D']:
            counts['D'] += 1
        if sample['A_given_C_eq_1']:
            counts['A_given_C_eq_1'] += 1

    fractions = {key: counts[key] / num_samples for key in counts}

    #L_1
    p_A_given_C = counts_AC / counts['C']

    #L_2
    p_A_do_C = fractions['A']

    #L_3
    P_A_C1_given_A_C = fractions['A_given_C_eq_1'] / fractions['A']

    results.append({'Trial': i+1, 'P(A=1)': fractions['A'], 'P(B=1)': fractions['B'],
                    'P(C=1)': fractions['C'], 'P(D=1)': fractions['D'], 'P(A=1|C=1)': p_A_given_C,
                    'P(A=1|do(C=1))': p_A_do_C, 'P(A_C=1=1|A=1,C=1)': P_A_C1_given_A_C})

df = pd.DataFrame(results)
print(df)
df.to_csv('results.csv', index=False)


# Create a line plot of the 'P(A=1)' column
plt.plot(df['Trial'], df['P(A=1|do(C=1))'])

# Add a horizontal line for the mean value
mean = df['P(A=1|do(C=1))'].mean()
minimum = df['P(A=1|do(C=1))'].min()
maximum = df['P(A=1|do(C=1))'].max()
plt.axhline(y=mean, color='r', linestyle='--')
plt.axhline(y=minimum, color='g', linestyle='--')
plt.axhline(y=maximum, color='b', linestyle='--')

print(minimum, mean, maximum)
# Add axis labels and a title
plt.xlabel('Trial')
plt.ylabel('P(A=1|do(C=1))')
plt.title('Probability of P(A=1|do(C=1)) over 20 Trials')

# Show the plot
plt.show()
plt.savefig('my_plot.png')

# # Define the variables to intervene on
# interventions = {'C': 1}
# numerator = compute_prob_given_parents('A', ['B', 'C', 'D'], interventions)
# denominator = sum(compute_prob_given_parents('A', ['B', 'C', 'D'], interventions.update({'A': a})) for a in [0, 1])
# print(numerator / denominator)

# Compute P(A=1 | do(C=1))
# print(compute_conditional_prob(interventions))