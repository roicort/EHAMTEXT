#!/bin/bash

left_hf_dataset=npvinHnivqn/EnglishDictionary
right_hf_dataset=npvinHnivqn/EnglishDictionary
left_hf_split=train
right_hf_split=train
left_col=word
right_col=definition
runpath=runs

export EHAM_LEFT_HF_DATASET="$left_hf_dataset"
export EHAM_RIGHT_HF_DATASET="$right_hf_dataset"
export EHAM_LEFT_HF_SPLIT="$left_hf_split"
export EHAM_RIGHT_HF_SPLIT="$right_hf_split"
export EHAM_LEFT_VIEW="$left_col"
export EHAM_RIGHT_VIEW="$right_col"
export EHAM_LEFT_COLUMN="$left_col"
export EHAM_RIGHT_COLUMN="$right_col"

echo "EAM Hetero text experiments."
echo "Left dataset: $left_hf_dataset ($left_hf_split)"
echo "Right dataset: $right_hf_dataset ($right_hf_split)"
echo "Columns: $left_col -> $right_col"
echo "Storing results in $runpath"
echo "=================== Starting at `date`"

# -f Genera los features: Actualmente usa SONAR 
# -c Por ahora no funciona ¿Cómo generamos prototipos ahora? ¿K-Means? ¿Qué K Usamos? 
# -s Evalua las memorias por dataset
# -e Evaluacion hetero
# -q Recall with cue
# -r Recall with sample and search
# -P Recall with prototype # Por ahora no funciona, no tenemos prototipos
# -p Recall with prototype and search # Por ahora no funciona, no tenemos prototipos
# -u Generate sequences

#uv run python eam.py -f --runpath=$runpath && \
##### uv run python eam.py -c --runpath=$runpath && \
#uv run python eam.py -s --runpath=$runpath && \
#uv run python eam.py -e --runpath=$runpath && \
#uv run python eam.py -q --runpath=$runpath && \
#uv run python eam.py -r --runpath=$runpath && \
###### uv run python eam.py -P constructed --runpath=$runpath && \
###### uv run python eam.py -P recalled --runpath=$runpath && \
###### uv run python eam.py -P extracted --runpath=$runpath && \
###### uv run python eam.py -p constructed --runpath=$runpath && \
###### uv run python eam.py -p recalled --runpath=$runpath && \
###### uv run python eam.py -p extracted --runpath=$runpath && \
uv run python eam.py -u --runpath=$runpath && \
echo "=================== Ending at `date`"
ok=$?
if [ $ok -eq 0 ]; then
    echo "Done."
else
    echo "Sorry, something went wrong."
fi