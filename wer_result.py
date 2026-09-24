# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import re
import sys
import argparse

try:
    from compute_wer import Calculator
except ImportError:
    print("Error: `compute_wer` package not found. Please ensure it is installed: pip install compute-wer>=0.2.1")
    sys.exit(1)

HALLUCINATION_NEW_HYP = "UNK"
WORD_RATIO_THRESHOLD = 7.0
PATH_WORD_NORM_SED_TXT = "word_norm_sed.txt"

DEFAULT_SED_WORD_NORM = r"""
s/$/ /g
s/<BABBLE>//g
s/<BABBLES>//g
s/<BABLE>//g
s/<NSN>//g
s/<SPN>//g
s/<BBBLE>//g
s/<LAUGH>//g
s/<LAUGHER>//g
s/<LAUGHS>//g
s/<LAUGHTER>//g
s/<OKAY>//g
s/<SIL>//g
s/<NON-SPEECH>//g
s/<FILLER>//g
s/\bFINANCIAL FORCE\b/ FINANCIALFORCE /g
s/\bSTACK OVERFLOW\b/ STACKOVERFLOW /g
s/\bSALES FORCE\b/ SALESFORCE /g
s/\bTARADATA\b/ TERADATA /g
s/\bTERRADATA\b/ TERADATA /g
s/\bRUBIK\b/ RUBRIK /g
s/\bRUBRIC\b/ RUBRIK /g
s/\bRUBICKS\b/ RUBRIKS /g
s/\bRUBRICK\b/ RUBRIK /g
s/\bRUBRIKK\b/ RUBRIK /g
s/<UNK>//g
s/<FILLER>//g
s/<FILLER>/<EPS>/g
s/\bAAH\b//g
s/\bHMM\b//g
s/\bAHM\b//g
s/\bUMM\b//g
s/\bUM\b//g
s/\bAAHM\b//g
s/\bUM\-HUM\b//g
s/\bUH\-HUH\b//g
s/\bUH\b//g
s/\bHUH\b//g
s/\bMM\b//g
s/\bMMM\b//g
s/\bMMMM\b//g
s/\bHMMM\b//g
s/\bHMM\b//g
s/\bAH\b//g
s/\bAAH\b//g
s/\bAHH\b//g
s/\bAHA\b//g
s/\bAAH\b//g
s/\bAAHH\b//g
s/\bEH\b//g
s/\bHA\b//g
s/\bHM\b//g
s/\bHO\b//g
s/\bOH\b//g
s/\bUH\b//g
s/\bUH-HUH\b//g
s/\bUM-HUM\b//g
s/\bUM-HUM\b//g
s/\bHUH\b//g
s/\bOOH\b//g
s/*//g
s/\bWE'RE\b/ WE ARE /g
s/\bI'M\b/ I AM /g
s/\bYOU'RE\b/ YOU ARE /g
s/\bDON'T\b/ DO NOT /g
s/\bWE'VE\b/ WE HAVE /g
s/\bI'LL\b/ I WILL /g
s/\bWE'LL\b/ WE WILL /g
s/\bTHEY'RE\b/ THEY ARE /g
s/\bI'VE\b/ I HAVE /g
s/\bYOU'VE\b/ YOU HAVE /g
s/\bI'D\b/ I WOULD /g
s/\bDIDN'T\b/ DID NOT /g
s/\bDOESN'T\b/ DOES NOT /g
s/\bWE'D\b/ WE WOULD /g
s/\bHAVEN'T\b/ HAVE NOT /g
s/\bWOULDN'T\b/ WOULD NOT /g
s/\bYOU'D\b/ YOU WOULD /g
s/\bISN'T\b/ IS NOT /g
s/\bTHAT'LL\b/ THAT WILL /g
s/\bYOU'LL\b/ YOU WILL /g
s/\bWASN'T\b/ WAS NOT /g
s/\bAREN'T\b/ ARE NOT /g
s/\bWEREN'T\b/ WERE NOT /g
s/\bIT'LL\b/ IT WILL /g
s/\bTHEY'VE\b/ THEY HAVE /g
s/\bTHEY'LL\b/ THEY WILL /g
s/\bSHOULDN'T\b/ SHOULD NOT /g
s/\bCOULDN'T\b/ COULD NOT /g
s/\bTHAT'D\b/ THAT WOULD /g
s/\bHASN'T\b/ HAS NOT /g
s/\bIT'D\b/ IT WOULD /g
s/\b'CAUSE\b/ BECAUSE /g
s/\b<INAUDIBLE>\b/ /g
s/\b<NOISE>\b/ /g
s/\b\[FILLER\/\]\b/ /g
s/\b\[N_S\/\]\b/ /g
s/\bA B C\b/ ABC /g
s/\bA M\b/ AM /g
s/\bA T M\b/ ATM /g
s/\bA. M.\b/ AM /g
s/\bACCOUNTS\b/ ACCOUNT /g
s/\bADDRESSES\b/ ADDRESS /g
s/\bADVICE\b/ ADVISE /g
s/\bAL RIGHT\b/ ALRIGHT /g
s/\bALL RIGHT\b/ ALRIGHT /g
s/\bANY MORE\b/ ANYMORE /g
s/\bANY WAY\b/ ANYWAY /g
s/\bAPOLOGISE\b/ APOLOGIZE /g
s/\bAUTHORISE\b/ AUTHORIZE /g
s/\bAUTHORISED\b/ AUTHORIZED /g
s/\bAUTO PAY\b/ AUTOPAY /g
s/\bAW\b/ /g
s/\bBE CAUSE\b/ BECAUSE /g
s/\bBILLS\b/ BILL /g
s/\bCALLS\b/ CALL /g
s/\bCAN\b/ COULD /g
s/\bCANCELED\b/ CANCELLED /g
s/\bCANCELING\b/ CANCELLING /g
s/\bCARDS\b/ CARD /g
s/\bCATHERIN\b/ CATHERINE /g
s/\bCHARGE BACK\b/ CHARGEBACK /g
s/\bCLAIMS\b/ CLAIM /g
s/\bCLEAR PAY\b/ CLEARPAY /g
s/\bCOMPLAIN\b/ COMPLAINT /g
s/\bCOSTS\b/ COST /g
s/\bCUSTOMERS\b/ CUSTOMER /g
s/\bDAYS\b/ DAY /g
s/\bDEANNA\b/ DIANA /g
s/\bDEDUCTIBLE\b/ DEDUCTABLE /g
s/\bDETAILS\b/ DETAIL /g
s/\bDIGITS\b/ DIGIT /g
s/\bDISCOUNTS\b/ DISCOUNT /g
s/\bDOLLARS\b/ DOLLAR /g
s/\bE MAIL\b/ EMAIL /g
s/\bELEVENTH\b/ ELEVEN /g
s/\bFEELS\b/ FEEL /g
s/\bFORTYFIVE\b/ FORTY FIVE /g
s/\bFOURTY\b/ FORTY /g
s/\bG B\b/ GB /g
s/\bG B I\b/ GBI /g
s/\bGIMME\b/ GIVE ME /g
s/\bGOIN'\b/ GOING /g
s/\bGONNA\b/ GOING TO /g
s/\bGOOD BYE\b/ GOODBYE /g
s/\bH D F C\b/ HDFC /g
s/\bH F D C\b/ HFDC /g
s/\bHA\b/ /g
s/\bHYDERABA\b/ HYDERABAD /g
s/\bI B C\b/ IBC /g
s/\bI P\b/ IP /g
s/\bI'D\b/ I WOULD /g
s/\bI'LL\b/ I WILL /g
s/\bI'M\b/ I AM /g
s/\bI'VE\b/ I HAVE /g
s/\bI. D.\b/ ID /g
s/\bI B\b/ IB /g
s/\bI C\b/ IC /g
s/\bI D\b/ ID /g
s/\bIN TO\b/ INTO /g
s/\bINCONIENCE\b/ INCONVENIENCE /g
s/\bINFORMATIONS\b/ INFORMATION /g
s/\bINQUIRE\b/ ENQUIRE /g
s/\bINQUIRING\b/ ENQUIRING /g
s/\bINQUIRY\b/ ENQUIRY /g
s/\bIT'S\b/ IT IS /g
s/\bITEMS\b/ ITEM /g
s/\bJ P\b/ JP /g
s/\bJ P MORGAN\b/ JPMORGAN /g
s/\bKINDA\b/ KIND A /g
s/\bLEMME\b/ LET ME /g
s/\bLILY\b/ LILLY /g
s/\bLOCATIONS\b/ LOCATION /g
s/\bLOG IN\b/ LOGIN /g
s/\bMA AM\b/ MAM /g
s/\bMA'AM\b/ MAM /g
s/\bMAAM\b/ MAM /g
s/\bMADAM\b/ MAM /g
s/\bMAY BE\b/ MAYBE /g
s/\bMICHEL\b/ MICHELLE /g
s/\bMM\b/ /g
s/\bMONTHS\b/ MONTH /g
s/\bMR\b/ MISTER /g
s/\bNAME'S\b/ NAME IS /g
s/\bNEAR BY\b/ NEARBY /g
s/\bNINTEEN\b/ NINETEEN /g
s/\bNINTY\b/ NINETY /g
s/\bNONFAT\b/ NON FAT /g
s/\bNONRECURRING\b/ NON RECURRING /g
s/\bNUMBERS\b/ NUMBER /g
s/\bOK\b/ OKAY /g
s/\bON LINE\b/ ONLINE /g
s/\bON TO\b/ ONTO /g
s/\bOPTIONS\b/ OPTION /g
s/\bORGANISATION\b/ ORGANIZATION /g
s/\bORGANISATIONS\b/ ORGANIZATIONS /g
s/\bOW\b/ /g
s/\bP M\b/ PM /g
s/\bPASS WORD\b/ PASSWORD /g
s/\bPAYMENTS\b/ PAYMENT /g
s/\bPER CENT\b/ PERCENT /g
s/\bPHONES\b/ PHONE /g
s/\bPLANS\b/ PLAN /g
s/\bPOINTS\b/ POINT /g
s/\bPOLICIES\b/ POLICY /g
s/\bPOSTCODE\b/ POST CODE /g
s/\bPOUND\b/ POUNDS /g
s/\bPREFILLED\b/ PRE FILLED /g
s/\bPRIZE\b/ PRICE /g
s/\bPROBLEMS\b/ PROBLEM /g
s/\bPROCEDURES\b/ PROCEDURE /g
s/\bPRODUCTS\b/ PRODUCT /g
s/\bPROGRAMME\b/ PROGRAM /g
s/\bQUESTIONS\b/ QUESTION /g
s/\bRATES\b/ RATE /g
s/\bRECOGNISE\b/ RECOGNIZE /g
s/\bRECORDS\b/ RECORD /g
s/\bRIA\b/ RIYA /g
s/\bSARA\b/ SARAH /g
s/\bSET UP\b/ SETUP /g
s/\bSEVENTYEIGHT\b/ SEVENTY EIGHT /g
s/\bSHORTTERM\b/ SHORT TERM /g
s/\bSHOULD'VE\b/ SHOULD HAVE /g
s/\bSIXTH\b/ SIX /g
s/\bSOME TIME\b/ SOMETIME /g
s/\bSTORES\b/ STORE /g
s/\bSURVEYS\b/ SURVEY /g
s/\bT V\b/ TV /g
s/\bTELECO\b/ TELCO /g
s/\bTENANT'S\b/ TENANTS /g
s/\bTHAT'D\b/ THAT WOULD /g
s/\bTHAT'LL\b/ THAT WILL /g
s/\bTHAT'S\b/ THAT IS /g
s/\bTHERE'S\b/ THERE IS /g
s/\bTHIRTYFOUR\b/ THIRTY FOUR /g
s/\bTRADEIN\b/ TRADE IN /g
s/\bTRAVELING\b/ TRAVELLING /g
s/\bTWELFTH\b/ TWELVE /g
s/\bTWENTYONE\b/ TWENTY ONE /g
s/\bTWENTYSIX\b/ TWENTY SIX /g
s/\bTWENTYTHREE\b/ TWENTY THREE /g
s/\bU K\b/ UK /g
s/\bU S\b/ US /g
s/\bUM\b/ /g
s/\bUP TO\b/ UPTO /g
s/\bWANNA\b/ WANT TO /g
s/\bWARRANT\b/ WARRANTY /g
s/\bWARRANTS\b/ WARRANTY /g
s/\bWE'LL\b/ WE WILL /g
s/\bWE'RE\b/ WE ARE /g
s/\bWE'VE\b/ WE HAVE /g
s/\bWHAT'S\b/ WHAT IS /g
s/\bWHATSAP\b/ WHATSAPP /g
s/\bWI FI\b/ WIFI /g
s/\bWI-FI\b/ WIFI /g
s/\bWILL\b/ WOULD /g
s/\bYADA\b/ YADAV /g
s/\bYEAH\b/ YES /g
s/\bYEP\b/ YES /g
s/\bYOU'D\b/ YOU WOULD /g
s/\bYOU'LL\b/ YOU WILL /g
s/\bYOU'RE\b/ YOU ARE /g
s/\bYOU'VE\b/ YOU HAVE /g
s/\bNYQUI\b/ NEXIUM /g
"""

def load_word_normalizations(word_norm_content):
    """Load sed-style normalization rules from content string or file."""
    normalizations = []

    # Regex to parse sed commands: s/pattern/replacement/flags
    # Matches unescaped / delimiters
    sed_pattern = re.compile(r'^s/((?:[^\\/]|\\.)*)/((?:[^\\/]|\\.)*)/(.*)$')

    # Split content into lines
    lines = word_norm_content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        match = sed_pattern.match(line)
        if match:
            pattern, replacement, flags = match.groups()
            normalizations.append({
                'pattern': f"\\{pattern}" if pattern == "*" else pattern,
                'replacement': replacement,
                'global': 'g' in flags
            })

    return normalizations


def apply_normalizations(text, normalizations):
    """Apply all normalization rules to the text."""
    for norm in normalizations:
        if norm['global']:
            text = re.sub(norm['pattern'], norm['replacement'], text)
        else:
            text = re.sub(norm['pattern'], norm['replacement'], text, count=1)
    return text


def read_and_normalize_kaldi_ark(file_path, normalizations):
    """Read Kaldi ark,t format file, normalize in memory, and return dict of utterance_id -> text."""
    utterances = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                utt_id, text = line.split('\t', 1)
                # Apply normalizations to text
                normalized_text = apply_normalizations(text, normalizations)
                utterances[utt_id] = normalized_text
            elif line:  # Handle space-separated format
                parts = line.split(None, 1)
                if len(parts) == 2:
                    normalized_text = apply_normalizations(parts[1], normalizations)
                    utterances[parts[0]] = normalized_text
                elif len(parts) == 1:
                    utterances[parts[0]] = ""
    return utterances


def fix_hallucination(text, utt_id):
    """Fix hallucination in prediction text."""
    words = text.split()
    if len(words) > 5 and len(words) / len(set(words)) > WORD_RATIO_THRESHOLD:
        return HALLUCINATION_NEW_HYP
    return text


def compute_wer_from_utterances(gt_utterances, pred_utterances, remove_hallucinations=True):
    """Compute WER using Calculator class from utterance dictionaries."""
    calculator = Calculator()

    # Calculate WER for each utterance
    for utt_id in gt_utterances:
        reference = gt_utterances.get(utt_id, "")
        hypothesis = pred_utterances.get(utt_id, "")

        # Remove hallucinations if requested
        if remove_hallucinations:
            hypothesis = fix_hallucination(hypothesis, utt_id)

        calculator.calculate(reference, hypothesis)

    # Get overall results
    overall_wer, _ = calculator.overall()

    return f"Overall WER: {overall_wer}\nOverall SER: {calculator.ser}\n"


def compute_wer(path, save_to_file=False, file_name="WER.txt"):
    """
    Compute WER from ground truth and prediction files.
    All processing is done in memory without creating intermediate files.

    Args:
        path: Path to the files (with or without _gt/_pred suffix)
        save_to_file: If True, save WER results to WER.txt file
    """
    # Get the directory of the script for finding word_norm.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    word_norm_file = os.path.join(script_dir, PATH_WORD_NORM_SED_TXT)
    
    # Load word normalization content from file or use default
    if os.path.exists(word_norm_file):
        print(f"Loading word normalizations from {word_norm_file}...")
        with open(word_norm_file, 'r') as f:
            word_norm_content = f.read()
    else:
        print(f"File {word_norm_file} not found, using default normalizations...")
        word_norm_content = DEFAULT_SED_WORD_NORM

    # Strip _gt or _pred suffix if present
    if path.endswith('_gt'):
        path = path[:-3]
    elif path.endswith('_pred'):
        path = path[:-5]

    # Define file paths
    gt_file = f"{path}_gt"
    pred_file = f"{path}_pred"

    # Check if files exist
    if not os.path.exists(gt_file):
        print(f"Error: Ground truth file {gt_file} not found")
        sys.exit(1)
    if not os.path.exists(pred_file):
        print(f"Error: Prediction file {pred_file} not found")
        sys.exit(1)

    # Parse normalization rules
    normalizations = load_word_normalizations(word_norm_content)
    print(f"Loaded {len(normalizations)} normalization rules")

    # Read and normalize files in memory
    print(f"Reading and normalizing {gt_file}...")
    gt_utterances = read_and_normalize_kaldi_ark(gt_file, normalizations)
    print(f"Loaded {len(gt_utterances)} ground truth utterances")

    print(f"Reading and normalizing {pred_file}...")
    pred_utterances = read_and_normalize_kaldi_ark(pred_file, normalizations)
    print(f"Loaded {len(pred_utterances)} prediction utterances")

    # Compute WER with hallucination removal
    print(f"\n=== Computing WER ===")
    wer_output_final = compute_wer_from_utterances(gt_utterances, pred_utterances)
    print(wer_output_final)
    
    # Save to file if requested
    if save_to_file:
        # Get the directory where the transcript files are located
        output_dir = os.path.dirname(gt_file)
        wer_file_path = os.path.join(output_dir, file_name)
        
        with open(wer_file_path, 'w') as f:
            f.write(wer_output_final)
        
        print(f"WER results saved to: {wer_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute WER (Word Error Rate) from ground truth and prediction files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python wer.py /path/to/results
  python wer.py /path/to/results_gt
  python wer.py /path/to/results --save
        '''
    )
    
    parser.add_argument(
        'path',
        help='Path to the files (with or without _gt/_pred suffix)'
    )
    
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Save WER results to WER.txt file in the same directory as the transcripts'
    )
    
    args = parser.parse_args()
    compute_wer(args.path, save_to_file=args.save)
