import glob
import pandas as pd
import os
# %%

for o in range(4,7):
    transcript_files = glob.glob("/home/sarulab/masaki_sato/デスクトップ/speech/NoLabel/transcript/transcript_JLecSponSpeech_Yoshimi_"+str(o)+".txt")
    # %%
    if not os.path.exists("raw_data/JLec_Yoshimi/JLec_Yoshimi"):
        os.makedirs("raw_data/JLec_Yoshimi/JLec_Yoshimi")
    for transcript in transcript_files:
        with open(transcript, mode='r') as f:
            lines = f.readlines()
        for line in lines:
            filename, text = line.split(':')
            with open('raw_data/JLec_Yoshimi/JLec_Yoshimi/' + filename + '.lab', mode='w') as f:
                f.write(text.strip('\n'))
