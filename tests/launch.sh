ds_leger_moyen=(speech cancers satellite shuttle fraude)
ds_leger=(speech cancers satellite shuttle)
ds_enfer=(http donneurs)

  
algo_sans_knn=(iforest lof ae vae deepsvdd ae1svm)

#python3 benchmark.py -d "${ds_leger_moyen[@]}" -a "${algo_sans_knn[@]}" 
#python3 benchmark.py -d "${ds_enfer[@]}" -a "${algo_sans_knn[@]}" 


python3 benchmark.py -d "${ds_leger[@]}" -a knn 
python3 benchmark.py -d fraude -a knn 


