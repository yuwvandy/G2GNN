echo "--no--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='no' --num_train=300 --num_val=300 --epochs=1000 --weight_decay=0.0005


echo "--upsampling--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='upsampling' --num_train=300 --num_val=300 --epochs=1000 --weight_decay=0.1


echo "--batch_reweight--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='batch_reweight' --num_train=300 --num_val=300 --epochs=1000 --weight_decay=0.0001


echo "--overall_reweight--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='overall_reweight' --num_train=300 --num_val=300 --epochs=1000 --weight_decay=0



echo "--smote--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='smote' --num_train=300 --num_val=300 --epochs=1000 --weight_decay=0.0001


echo "--knn--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='knn' --num_train=300 --num_val=300 --epochs=1000 --weight_decay=0.05 --knn_layer=3 --knn_nei_num=2


echo "--aug_drop_edge--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='aug' --num_train=300 --num_val=300 --epochs=1000 --weight_decay=0.01 --drop_edge_ratio=0.02


echo "--aug_mask_node--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='aug' --num_train=300 --num_val=300 --epochs=1000 --weight_decay=0.1 --mask_node_ratio=0.0005


echo "--knn_aug_drop_edge--"
python main.py --dataset='PROTEINS' --imb_ratio=0.9 --setting='knn_aug' --num_train=300 --num_val=300 --epochs=1000 --weight_decay=0.1 --knn_layer=1 --knn_nei_num=2 --drop_edge_ratio=0.005
