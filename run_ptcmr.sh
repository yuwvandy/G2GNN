echo "--no--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='no' --num_train=90 --num_val=90 --epochs=1000 --weight_decay=0.0005


echo "--upsampling--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='upsampling' --num_train=90 --num_val=90 --epochs=1000 --weight_decay=0.05


echo "--batch_reweight--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='batch_reweight' --num_train=90 --num_val=90 --epochs=1000 --weight_decay=0.001


echo "--overall_reweight--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='overall_reweight' --num_train=90 --num_val=90 --epochs=1000 --weight_decay=0.0005



echo "--smote--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='smote' --num_train=90 --num_val=90 --epochs=1000 --weight_decay=0.0005



echo "--knn--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='knn' --num_train=90 --num_val=90 --epochs=1000 --weight_decay=0.01 --knn_layer=3 --knn_nei_num=2



echo "--aug_drop_edge--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='aug' --num_train=90 --num_val=90 --epochs=1000 --weight_decay=0.1 --drop_edge_ratio=0.05


echo "--aug_mask_node--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='aug' --num_train=90 --num_val=90 --epochs=1000 --weight_decay=0.001 --mask_node_ratio=0.001


echo "--knn_aug_drop_edge--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='knn_aug' --num_train=90 --num_val=90 --epochs=1000 --weight_decay=0.01 --knn_layer=3 --knn_nei_num=3 --drop_edge_ratio=0.05


echo "--knn_aug_mask_node--"
python main.py --dataset='PTC_MR' --imb_ratio=0.9 --setting='knn_aug' --num_train=90 --num_val=90 --epochs=1000 --weight_decay=0.1 --knn_layer=2 --knn_nei_num=2  --mask_node_ratio=0.0005

