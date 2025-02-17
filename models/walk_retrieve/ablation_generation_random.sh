walk_distance=("1" "2" "3" "4" "5" "6")
walk_number=("10" "20" "30" "40" "50" "60" "70" "80" "90" "100")
walk_strategy=("random")

for walk in ${walk_strategy[*]}
do
    for distance in ${walk_distance[*]}
    do
        for number in ${walk_number[*]}
        do
            echo $walk $distance $number
            python models/walk_retrieve/kg_corpus_generation.py -p data/MetaQA/kb.parquet -d $distance -w $number -chunk 100 -save data/MetaQA/rdf_corpus -cpu 20 -walk $walk
        done        
    done
done