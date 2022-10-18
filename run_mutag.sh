echo "--no--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='no' --num_train=50 --num_val=50 --epochs=1000 --weight_decay=0.005


echo "--upsampling--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='upsampling' --num_train=50 --num_val=50 --epochs=1000 --weight_decay=0.01


echo "--batch_reweight--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='batch_reweight' --num_train=50 --num_val=50 --epochs=1000 --weight_decay=0.005


echo "--overall_reweight--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='overall_reweight' --num_train=50 --num_val=50 --epochs=1000 --weight_decay=0.05



echo "--smote--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='smote' --num_train=50 --num_val=50 --epochs=1000 --weight_decay=0.01


echo "--knn--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='knn' --num_train=50 --num_val=50 --epochs=1000 --weight_decay=0.05 --knn_layer=3 --knn_nei_num=2


echo "--aug_drop_edge--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='aug' --num_train=50 --num_val=50 --epochs=1000 --weight_decay=0.05 --drop_edge_ratio=0.005


echo "--aug_mask_node--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='aug' --num_train=50 --num_val=50 --epochs=1000 --weight_decay=0.1 --mask_node_ratio=0.1



echo "--knn_aug_drop_edge--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='knn_aug' --num_train=50 --num_val=50 --epochs=1000 --weight_decay=0.01 --knn_layer=2 --knn_nei_num=2 --drop_edge_ratio=0.001

echo "--knn_aug_mask_node--"
python main.py --dataset='MUTAG' --imb_ratio=0.1 --setting='knn_aug' --num_train=50 --num_val=50 --epochs=1000 --weight_decay=0.05 --knn_layer=2 --knn_nei_num=2  --mask_node_ratio=0.001
