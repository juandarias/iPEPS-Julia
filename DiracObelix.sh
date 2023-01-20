#!/bin/sh
#rsync -r -v  "$PWD/" "jdarias@obelix-h0.science.uva.nl:Code/Julia/iPEPS/"

rsync -r -v  "$PWD/src" "jdarias@obelix-h0.science.uva.nl:Code/Julia/iPEPS/"

rsync -r -v  "$PWD/input" "jdarias@obelix-h0.science.uva.nl:Code/Julia/iPEPS/"

#rsync -r -v  "$PWD/scripts" "jdarias@obelix-h0.science.uva.nl:Code/Julia/iPEPS/"


