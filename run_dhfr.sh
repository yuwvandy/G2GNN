echo "======DHFR======" 'imb_ratio' 0.1

echo "--no--"
python main.py --dataset='DHFR' --imb_ratio=0.1 --setting='no' --num_training=120 --num_val=120 --epochs=1000 --weight_decay=0

echo "--upsampling--"
python main.py --dataset='DHFR' --imb_ratio=0.1 --setting='upsampling' --num_training=120 --num_val=120 --epochs=1000 --weight_decay=0.0005

echo "--reweight--"
python main.py --dataset='DHFR' --imb_ratio=0.1 --setting='reweight' --num_training=120 --num_val=120 --epochs=1000 --weight_decay=0.0005

echo "--smote--"
python main.py --dataset='DHFR' --imb_ratio=0.1 --setting='smote' --num_training=120 --num_val=120 --epochs=1000 --weight_decay=0.0001


echo "--knn--"
python main.py --dataset='DHFR' --imb_ratio=0.1 --setting='knn' --num_training=120 --num_val=120 --epochs=1000 --weight_decay=0 --prop_epochs=2 --knn=3


echo "--AUG-RE--"
python main.py --dataset='DHFR' --imb_ratio=0.1 --setting='aug' --aug='RE' --num_training=120 --num_val=120 --epochs=1000 --weight_decay=0 --aug_ratio=0.01


echo "--AUG-DN--"
python main.py --dataset='DHFR' --imb_ratio=0.1 --setting='aug' --aug='DN' --num_training=120 --num_val=120 --epochs=1000 --weight_decay=0 --aug_ratio=0.01


echo "--knn_aug-RE--"
python main.py --dataset='DHFR' --imb_ratio=0.1 --setting='knn_aug' --aug='RE' --num_training=120 --num_val=120 --epochs=1000 --weight_decay=0  --aug_ratio=0.01 --prop_epochs=3 --knn=3


echo "--knn_aug-DN--"
python main.py --dataset='DHFR' --imb_ratio=0.1 --setting='knn_aug' --aug='DN' --num_training=120 --num_val=120 --epochs=1000 --weight_decay=0  --aug_ratio=0.01 --prop_epochs=3 --knn=3
