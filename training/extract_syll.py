import pandas as pd
import root.scripts.SYLL as SYLL

def extract_syll():
    data = pd.read_csv("root/datasets/FakeNewsPhilippines2024_Lupac.csv")

    try:
        syll_features = pd.read_csv("root/datasets/SyllFeatures.csv")
    except FileNotFoundError:
        columns = "consonant_cluster,v_density,cv_density,vc_density,cvc_density,vcc_density,cvcc_density,ccvcc_density,ccvccc_density"
        with open("root/datasets/SyllFeatures.csv", 'w') as f:
            f.write(columns)
        syll_features = pd.read_csv("root/datasets/SyllFeatures.csv")

    list_of_vals = [ ]
    starting_index = -1  # TODO: Check last progress of the corresponding .csv and change starting index as needed

    # consonant_cluster = SYLL.get_consonant_cluster(data['article'][0])
    # v_density = SYLL.get_v(data['article'][0])
    # cv_density = SYLL.get_cv(data['article'][0])
    # vc_density = SYLL.get_vc(data['article'][0])
    # cvc_density = SYLL.get_cvc(data['article'][0])
    # vcc_density = SYLL.get_vcc(data['article'][0])
    # cvcc_density = SYLL.get_cvcc(data['article'][0])
    # ccvcc_density = SYLL.get_ccvcc(data['article'][0])
    # ccvccc_density = SYLL.get_ccvccc(data['article'][0])

    # list_of_vals.append({
    # "consonant_cluster": consonant_cluster,
    # "v_density": v_density,
    # "cv_density": cv_density,
    # "vc_density": vc_density,
    # "cvc_density": cvc_density,
    # "vcc_density": vcc_density,
    # "cvcc_density": cvcc_density,
    # "ccvcc_density": ccvcc_density,
    # "ccvccc_density": ccvccc_density
    # })

    # syll_features = pd.concat([pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals)))), syll_features])
    # syll_features.to_csv("root/datasets/SyllFeatures.csv", index=False)

    # exit(0)
    
    for i in data.itertuples():
        if (i.Index > starting_index):
            consonant_cluster = SYLL.get_consonant_cluster(i.article)
            v_density = SYLL.get_v(i.article)
            cv_density = SYLL.get_cv(i.article)
            vc_density = SYLL.get_vc(i.article)
            cvc_density = SYLL.get_cvc(i.article)
            vcc_density = SYLL.get_vcc(i.article)
            cvcc_density = SYLL.get_cvcc(i.article)
            ccvcc_density = SYLL.get_ccvcc(i.article)
            ccvccc_density = SYLL.get_ccvccc(i.article)
            
            #   Append to list
            list_of_vals.append({
                "consonant_cluster": consonant_cluster,
                "v_density": v_density,
                "cv_density": cv_density,
                "vc_density": vc_density,
                "cvc_density": cvc_density,
                "vcc_density": vcc_density,
                "cvcc_density": cvcc_density,
                "ccvcc_density": ccvcc_density,
                "ccvccc_density": ccvccc_density
            })
            
            print(i.Index)
            
            if ((i.Index)%25) == 0:
                print("Saving figures...")
                syll_features = pd.concat([syll_features, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
                syll_features.to_csv("root/datasets/SyllFeatures.csv", index=False)
                list_of_vals = []
                
    syll_features = pd.concat([syll_features, pd.DataFrame(list_of_vals, index=list(range(len(list_of_vals))))])
    syll_features.to_csv("root/datasets/SyllFeatures.csv", index=False)

def main():
    extract_syll()
    
if __name__ == "__main__":
    main()