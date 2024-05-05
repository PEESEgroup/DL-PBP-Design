import numpy as np
import sys
#from copy import deepcopy

#default amino acid sequence for calculating solubility
sequence = 'AWTFHRK'
pep_length = len(sequence)

#### PARAMETERS #####

#amino acid parameters
#values in array are: net charge, hydrophobicity, alpha helix propensity, and beta helix propensity
AA_params = {'A': [0, -0.39, 0.79, 0.7], 
             'C': [0, 2.25, 0.49, 0.84], 
             'D': [-1, 3.81, 0.65, 0.68], 
             'E': [-1, 2.91, 0.97, 0.73], 
             'F': [0, -2.27, 0.77, 0.98], 
             'G': [0, 0.05, 0.21, 0.15], 
             'H': [0, 0.64, 0.8, 0.8], 
             'I': [0, -1.82, 0.93, 1.0], 
             'K': [1, 2.77, 0.88, 0.73], 
             'L': [0, -1.82, 0.98, 0.86], 
             'M': [0, -0.96, 0.82, 0.82], 
             'N': [0, 1.91, 0.61, 0.6], 
             'P': [0, -0.99, 0.17, 0.09], 
             'Q': [0, 1.3, 0.78, 0.77], 
             'R': [1, 3.5, 0.95, 0.79], 
             'S': [0, 1.24, 0.67, 0.67], 
             'T': [0, 1.0, 0.52, 0.86], 
             'V': [0, -1.3, 0.75, 1.05], 
             'W': [0, -2.13, 0.64, 0.86], 
             'Y': [0, -1.47, 0.73, 0.89]}

#amino acid polarity, as given here: https://doi.org/10.1006/jmbi.2000.3514 (this presumably is the classification used by CamSol, but I don't see this ever specifically stated)
# 1=polar, 2=non-polar, 0=neither
AA_polarity = {'A': 0,
               'C': 0,
               'D': 1,
               'E': 1,
               'F': 2,
               'G': 0,
               'H': 1,
               'I': 2,
               'K': 1,
               'L': 2,
               'M': 2,
               'N': 1,
               'P': 0,
               'Q': 1,
               'R': 1,
               'S': 0,
               'T': 0,
               'V': 2,
               'W': 0,
               'Y': 0}

#CamSol coefficients
#order is coefficients for hydrophobicity, charge, alpha-helix propensity, then beta-sheet propensity
#put in array format for notational convenience in si_individual function
CS_individual_params = np.array([.598, 0.318, 5.77, -4.807]) 

#Camsol parameters used to smooth profile over resiudes
CS_params = {'a_pat': -2.816,
             'a_gk' :0.152,
             'si_cutoff': 0.0}

#parameters to normalize intrinsic solubility values for individual amino acids (don't use these at the moment)
sigma = 0.28067173
mu =1.05122082

#### END OF PARAMETERS #####

#### AUXILIARY FUNCTIONS CALLED DURING CAMSOL CALCULATION #####

def si_individual_calc(pep_params):
    return np.matmul(CS_individual_params, pep_params)

def I_gk_calc(charges):
    I_gk = np.zeros((pep_length,))
    for i in range(pep_length):
        I_gk[i] = np.sum(charges[max(0,i-5):min(pep_length,i+5)] *   
             np.array([np.exp(-(j**4)/200) for j in range(max(0,i-5), min(pep_length, i+5))]))
    #for i in range(5, pep_length-5):
    #    I_gk[i] = np.sum(charges[i-5:i+5] * np.array([np.exp(-(j**4)/200) for j in range(i-5, i+5)]))
    return I_gk

def I_pat_calc(sequence):
    types = np.array([AA_polarity[AA] for AA in sequence])
    I_pat = np.zeros(pep_length)
    for i in range(2,pep_length-2):
        #do simple check for pattern: sum over 5 residue window needs to be either 7 or 8 for alternating hydrophobic/hydrophilic
        pattern_sum = sum(types[i-2:i+2])
        if pattern_sum == 7 or pattern_sum == 8:
            #now do more rigorous check to see if alternating pattern is found. 
            #Note that there are only two sets of sums possible to have pattern; if statement below checks if either of two sums is found
            pat1 = types[i-2]+types[i]+types[i+2]
            pat2 = types[i-1]+types[i+1]
            if (pat1 == 3 and pat2 == 4) or (pat1 == 6 and pat2 == 2):
                I_pat[i] = 1
    return I_pat

def si_smooth_calc(pep_params, si_individual):
    I_gk = I_gk_calc(pep_params[0,:])
    I_pat = I_pat_calc(sequence)
    Si_window = np.array([(1/(min(pep_length, i+3) - max(0, i-3))) * sum(si_individual[max(0, i-3):min(pep_length, i+3)]) for i in range(pep_length)])
    return Si_window + CS_params['a_pat']*I_pat + CS_params['a_gk']*I_gk

    ### use below if you want the smoothed, normalized intrinsic solubility profiles (typically don't need)
    #Si_window = deepcopy(si_individual)
    #Si_window[3:pep_length-3] = np.array([(1/7) * sum(si_individual[i-3:i+3]) for i in range(3,pep_length-3)])
    #return Si_window + CS_params['a_pat']*I_pat + CS_params['a_gk']*I_gk

def net_solubility_calc(si_smooth):
    si_smooth_masked = si_smooth[abs(si_smooth) > CS_params['si_cutoff']]
    return np.average(si_smooth_masked)

#### END AUXILIARY FUNCTIONS CALLED DURING CAMSOL CALCULATION #####

#### MAIN CAMSOL CALCULATION #####

def CamSol_calc(sequence):
    
    #parameters for peptide given its sequence
    charges = np.array([AA_params[AA][0] for AA in sequence])
    phobs = np.array([AA_params[AA][1] for AA in sequence])
    alphas = np.array([AA_params[AA][2] for AA in sequence])
    betas = np.array([AA_params[AA][3] for AA in sequence])
    
    #store values in matrix for ease of calculation
    pep_params = np.stack([charges,phobs, alphas, betas])

    #get initial amino acid si values, excluding neighboring residues
    si_individual = si_individual_calc(pep_params)
    
    #smooth si values based on amino acids nearby in sequence
    si_smooth = si_smooth_calc(pep_params, si_individual)
    
    #get net solubility of peptide
    net_solubility = net_solubility_calc(si_smooth)
    
    #return value - you did it!!
    return net_solubility
    
    ### use below if you want the smoothed, normalized intrinsic solubility profiles (typically don't need)

    #get normalized smoothed profiles
    #si_smooth_normalized = (si_smooth - mu)/sigma
    
    #return value - you did it!!
    #return net_solubility, si_smooth_normalized

#### END CAMSOL CALCULATION #####

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Calculating solubility of default peptide sequence!')
    elif len(sys.argv) > 2:
        print('Please provide exactly 1 amino acid sequence (no spaces, single letter codes)!')
        sys.exit()
    else:
        sequence = sys.argv[1]
        pep_length = len(sequence)
    
    solubility = CamSol_calc(sequence)
    print(f'Solubility: {solubility}')
    #print(f'Solubility profile: {si_smooth_normalized}')


