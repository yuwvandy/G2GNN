echo "======PTC-MR======" 'imb_ratio' 0.9
echo "--no--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='no' --num_training=90 --num_val=90 --epochs=1000 --weight_decay=0.0001

echo "--upsampling--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='upsampling' --num_training=90 --num_val=90 --epochs=1000 --weight_decay=0

echo "--reweight--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='reweight' --num_training=90 --num_val=90 --epochs=1000 --weight_decay=0.001

echo "--smote--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='smote' --num_training=90 --num_val=90 --epochs=1000 --weight_decay=0.0001

echo "--knn--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='knn' --num_training=90 --num_val=90 --epochs=1000 --weight_decay=0.01 --prop_epochs=1 --knn=2

echo "--AUG-RE--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='aug' --aug='RE' --num_training=90 --num_val=90 --epochs=1000 --weight_decay=0.01 --aug_ratio=0.1

echo "--knn_aug-RE--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='knn_aug' --aug='RE' --num_training=90 --num_val=90 --epochs=1000 --weight_decay=0.01  --aug_ratio=0.1 --prop_epochs=2 --knn=3

echo "--AUG-DN--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='aug' --aug='DN' --num_training=90 --num_val=90 --epochs=1000 --weight_decay=0.01 --aug_ratio=0.1

echo "--knn_aug-DN--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='knn_aug' --aug='DN' --num_training=90 --num_val=90 --epochs=1000 --weight_decay=0.01  --aug_ratio=0.1 --prop_epochs=2 --knn=3
