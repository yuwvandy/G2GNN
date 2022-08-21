echo "======MUTAG======" 'imb_ratio' 0.1
echo "--no--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='no' --num_training=50 --num_val=50 --epochs=1000 --weight_decay=0.0001

echo "--upsampling--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='upsampling' --num_training=50 --num_val=50 --epochs=1000 --weight_decay=0.01

echo "--reweight--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='reweight' --num_training=50 --num_val=50 --epochs=1000 --weight_decay=0.01

echo "--smote--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='smote' --num_training=50 --num_val=50 --epochs=1000 --weight_decay=0.1

echo "--knn--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='knn' --num_training=50 --num_val=50 --epochs=2000 --weight_decay=0.1 --prop_epochs=3 --knn=3

echo "--AUG-RE--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='aug' --aug='RE' --num_training=50 --num_val=50 --epochs=1000 --weight_decay=0.01 --aug_ratio=0.1

echo "--knn_aug-RE--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='knn_aug' --aug='RE' --num_training=50 --num_val=50 --epochs=1000 --weight_decay=0.05  --aug_ratio=0.1 --prop_epochs=3 --knn=3

echo "--AUG-DN--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='aug' --aug='DN' --num_training=50 --num_val=50 --epochs=1200 --weight_decay=0.001 --aug_ratio=0.2

echo "--knn_aug-DN--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='knn_aug' --aug='DN' --num_training=50 --num_val=50 --epochs=2000 --weight_decay=0.1  --aug_ratio=0.2 --prop_epochs=2 --knn=3
