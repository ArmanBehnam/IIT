import numpy as np

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
counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'A_given_C_eq_1': 0}
counts_AC = 0
results = {'U_A': 0, 'U_AB': 0, 'U_AC': 0, 'U_B': 0, 'U_C': 0, 'U_D': 0}

for i in range(num_samples):
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

    results['U_A'] += sample['U_A1'] + sample['U_A2']
    results['U_AB'] += sample['U_AB']
    results['U_AC'] += sample['U_AC']
    results['U_B'] += sample['U_B']
    results['U_C'] += sample['U_C']
    results['U_D'] += sample['U_D']

results['U_A'] /= num_samples
results['U_AB'] /= num_samples
results['U_AC'] /= num_samples
results['U_B'] /= num_samples
results['U_C'] /= num_samples
results['U_D'] /= num_samples

fractions = {key: counts[key] / num_samples for key in counts}

print(sample)
print("Fraction of samples that satisfy f_A: ", fractions['A'])
print("Fraction of samples that satisfy f_B: ", fractions['B'])
print("Fraction of samples that satisfy f_C: ", fractions['C'])
print("Fraction of samples that satisfy f_D: ", fractions['D'])


#L_1
p_A_given_C = counts_AC / counts['C']
print("P(A = 1 | C = 1):", p_A_given_C)

#L_2
print("Fraction of samples that satisfy f_A when C is intervened on and set to 1: ", fractions['A'])
print("P(A = 1 | do(C = 1)): ", fractions['A'])

#L_3
print("Fraction of samples that satisfy A given C=1: ", fractions['A_given_C_eq_1'])
P_A_C1_given_A_C = fractions['A_given_C_eq_1'] / fractions['A']
print("P(A_{C=1} = 1 | A = 1, C = 1):", P_A_C1_given_A_C)