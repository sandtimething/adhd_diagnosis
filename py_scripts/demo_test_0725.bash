read -p "Input the new caseid: " caseid

# run the whole dataset when algorithm is updated
# python percentile_witherrorhandling_2.py 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 41 42 43 44 45 46 47 49 50 51 52 61 62 63 64 65 66 67 68 $caseid
python percentile_witherrorhandling_2.py $caseid
echo percentile done

python rose_witherrorhandling.py $caseid
echo rose done

# run the whole dataset when algorithm is updated
# python signal_detection_vrclassroom.py 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 41 42 43 44 45 46 47 49 50 51 52 61 62 63 64 65 66 67 68 $caseid
python signal_detection_vrclassroom.py $caseid
echo signal_detection done

# python bayes_model_vrclassroom_noSnap.py 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 41 42 43 44 45 46 47 48 
# python bayes_model_vrclassroom_noSnap.py 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 61 62 63 64 65 66 67 68
python bayes_model_vrclassroom.py $caseid
echo bayes model done