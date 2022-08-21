echo "======PROTEINS======" 'imb_ratio' 0.9
echo "--no--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='no' --num_training=300 --num_val=300 --epochs=1000 --weight_decay=0.001

echo "--upsampling--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='upsampling' --num_training=300 --num_val=300 --epochs=1000 --weight_decay=0.1

echo "--reweight--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='reweight' --num_training=300 --num_val=300 --epochs=1000 --weight_decay=0.0001

echo "--smote--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='smote' --num_training=300 --num_val=300 --epochs=1000 --weight_decay=0.0001

echo "--knn--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='knn' --num_training=300 --num_val=300 --epochs=1000 --weight_decay=0.05 --prop_epochs=1 --knn=3

echo "--AUG-RE--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='aug' --aug='RE' --num_training=300 --num_val=300 --epochs=1000 --weight_decay=0.1 --aug_ratio=0.2

echo "--knn_aug-RE--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='knn_aug' --aug='RE' --num_training=300 --num_val=300 --epochs=1000 --weight_decay=0.05  --aug_ratio=0.1 --prop_epochs=2 --knn=4

echo "--AUG-DN--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='aug' --aug='DN' --num_training=300 --num_val=300 --epochs=1000 --weight_decay=0.1 --aug_ratio=0.2

echo "--knn_aug-DN--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='knn_aug' --aug='DN' --num_training=300 --num_val=300 --epochs=1000 --weight_decay=0.05  --aug_ratio=0.1 --prop_epochs=2 --knn=4
