### 统一符号说明
1. b: batch_size
2. fna: feature_map_num_anchors
3. fsp,fh,fw,fc:feature_map_shape,feature_map_height,feature_map_weight,feature_map_channel
4. nc: num_class,voc数据就是20类
5. pred_bdim: pred_box_dim,(nc + 1 + 4)
6. wh_dim: 2，量纲是pixel
7. fmr,fmgs: feature_map_resolution,feature_map_grid_size
8. fm_anchors: feature_map_anchors
9. h,w,c: image_shape (512,512,3)
10. bx,by,bw,bh: box_center_pu,box_center_pv,box_weight,box_height
11. ubx,uby,ubw,ubh: unit_box_center_pu,unit_box_center_pv,unit_box_weight,unit_box_height
12. fbx,fby,fbw,fbh: feature_map_box_center_pu,feature_map_box_center_pv,feature_map_box_weight,feature_map_box_height
13. nt: num_target
14. nl: num_layers,3
15. td: target_dim 
    1.  targets: (img_idx,lable_idx,ubx,uby,ubw,ubh)
    2.  fm_targets: feature_map_targets,(img_idx,lable_idx,fbx,fby,fbw,fbh)
16. feature map: (b,fna,fh,fw,bdim)
17. ecd_dim: encoder_dim:　其实就是９个值，４个索引，５个数值
    1. img_idx: image_index (0~batch_size)
    2. fan_idx: feature_map_anchor_index (0~fna)
    3. fgh_idx: feature_map_grid_hidx (0~fh)
    4. fgw_idx: feature_map_grid_widx (0~fw)
    5. gt_box_dim: 5个参数就够了: (lable,fsux,fsuy,gtw,gth)
       1. lable: (0~num_class)
       2. fsux,fsuy: feature_map_shift_unit_pu,feature_map_shift_unit_pv (0~1)
       3. gtw,gth: gt_weight,gt_height 量纲是pixel
18. pred_box_dim: 25个参数,x,y,w,h,c0,c1,c2,c3...,c20
    1. x,y: 
       1. 数值随便,要监督的时候做sigmoid: sigx,sigy,(0~1)
       2. $$x_1,y_1 \xrightarrow{\sigma(x)} x_2,y_2 \xrightarrow{(+grid)\times fmr}u,v(pixel)$$
    2. w,h: 
       1. 数值随便,在转为wh的时候会 1.exp 2.乘以anchors 所以梯度很大,取0的时候都是1倍了,应该是差不多就是在0的附近
       2. $$w_1,h1 \xrightarrow{e^x} w_2,h_2 \xrightarrow{\times an\_wh}$$ w_3,h_3(pixel)
    3. c0,c1,...,c20
       1. 数值随便,最后会经过一个softmax层压下来 
       2. $$c_0,c_1,...,c_{20} \xrightarrow{softmax} s_0,s_1,...,s_{20}$$
### targets(Dless) vs box(pixel)
#### targets+image_shape->box
bx = w x ubx
by = h x uby
bw = w x ubw
bh = h x ubh
#### targets+feature_shape->feature_map_box
fbx = fw x ubx
fby = fh x uby
fbw = fw x ubw
fbh = fh x ubh
#### 数值
```python
tensor([[ 0.00000,  8.00000,  0.54148,  0.20321,  0.24600,  0.27800],
        [ 0.00000,  8.00000,  0.64448,  0.47621,  0.34800,  0.28400],
        [ 0.00000,  4.00000,  0.13424,  0.23910,  0.26848,  0.47821],
        [ 0.00000, 10.00000,  0.20824,  0.30910,  0.41648,  0.61821],
        [ 0.00000, 12.00000,  0.81024,  0.83210,  0.37952,  0.33579],
        [ 0.00000,  3.00000,  0.20824,  0.81010,  0.41648,  0.37979],
        [ 1.00000,  2.00000,  0.37974,  0.22076,  0.57400,  0.31589],
        [ 1.00000,  2.00000,  0.42074,  0.88931,  0.50400,  0.22138],
        [ 2.00000, 14.00000,  0.91834,  0.71675,  0.12200,  0.50400],
        [ 2.00000, 11.00000,  0.46734,  0.63987,  0.61600,  0.34188],
        [ 3.00000, 16.00000,  0.68831,  0.19823,  0.62338,  0.39647],
        [ 3.00000, 16.00000,  0.92331,  0.33255,  0.15338,  0.23568],
        [ 3.00000, 16.00000,  0.73762,  0.34554,  0.16200,  0.13382],
        [ 3.00000, 16.00000,  0.67931,  0.48934,  0.64138,  0.44939],
        [ 3.00000, 19.00000,  0.45762,  0.94702,  0.19400,  0.10597]])
```
### 三层的preds，三层feature map
#### 第一层,fmr最大,fmgs最少
1. feature map: fm输出的所有数字
   1. (b,fna,fh,fw,pred_bdim): 4 x 3 x 16 x 16 x 25
2. fmr
   1. H / FH = W / FW = 32
3. fm_anchors: fm中需要预测目标
   1. (fna x wh_dim): 3 x 2
      1. [[116, 90], [156, 198], [373, 326]]
#### 第二层,fmr变小,fmgs变大
1. feature map shape: 4 x 3 x 32 x 32 x 25 
2. fmr: 16
3. fm_anchors: [[30, 61], [62, 45], [59, 119]]
####　第三层，fmr最小，fmgs最多
1. feature map shape: 4 x 3 x 64 x 64 x 25 
2. fmr: 8
3. fm_anchors: [[10, 13], [16, 30], [33, 23]]
### 后两层feature map的获取
#### 上采样pixel2: v2的向量原来都是相互独立的，通过768x256得到的最后的256: v3,v3相当于v1和v2 fusion的结果
### encoder and decoder
#### encoder: targets+fm_anchors -> loss
1. input
   1. targets: (nt,td)
   2. fm_anchors: (fna,wh_dim)
   3. layer_pred: (b,fna,fh,fw,pred_bdim)
   4. fmr
2. output
   1. loss (1,)
3. encoder process
   1. 匹配分配给该层的fm_anchors和targets
      1. input: fm_anchors,fmr,targets
      2. output: mask: (fna,nt)
      3. process
         1. fm_targets: (nt,td)
            1. targets + fmr
         2. mask: (fna,nt),type: bool
            1. fm_anchors vs fm_targets
         3. layer_labels: (fna,nt,ecd_dim)获取９个参数
            1. img_idx＋fan_idx＋fgh_idx＋fgw_idx＋gt_box_dim
         4. loss: layer_pred vs layer_labels
            1. layer_pred: (b,fna,fh,fw,pred_bdim)
            2. layer_labels: (fna,nt,ecd_dim)
            3. for gt_box in layer_labels:
               1. idx,tbox =[img_idx,fan_idx,fgh_idx,fgw_idx](4,),[lable,fsux,fsuy,gtw,gth](5,)
               2. pbox = layer_pred[img_idx,fan_idx,fgh_idx,fgw_idx] (25,)
               3. anchor = fm_anchors[fan_idx] (2,)
                  1. pxy = pbox[:2].sigmoid()
                  2. pwh = pbox[2:4].exp()*anchor
                     1. loss_giou = [fsux,fsuy,gtw,gth] vs [pxy,pwh]
                     2. loss_lable = [c0,c1,...,c20] vs lable
                     3. loss_obj = softmax(c0) vs (0 or 1)
#### decoder: preds解码出bboxs
1. input: preds
   1. (1,fna,fh,fw,pred_bdim)
      1. pred1 = torch.Size([1, 3, 16, 16, 25]): torch.Size([1, 768, 25])
      2. pred2 = torch.Size([1, 3, 32, 32, 25]): torch.Size([1, 3072, 25])
      3. pred3 = torch.Size([1, 3, 64, 64, 25]): torch.Size([1, 12288, 25])
2. output: (n,box_dim): x,y,w,h,lable,prob
3. process: 针对pred,torch.Size([1, 3, 16, 16, 25])
   1. x,y: pred[...,:2]
      1. sigmoid
      $$x_1,y_1 \xrightarrow{\sigma(x)} x_2,y_2$$
      2. grid 
         ``` python
         xx,yy = torch.meshgrid(torch.arange(h),torch.arange(w))
         grid = torch.stack((xx,yy),dim=-1)
         grid = grid.view((1,1,h,w,2)).type(torch.float)
         ```
      3. pred->pixels
      $$pred \xrightarrow{(\sigma(x)+grid)\times stride}pred（pixel） $$
   1. w,h: pred[...,2:4]
      1. exp
      $$w_1,h_1 \xrightarrow{e^x} w_2,h_2$$
      2. pred->pixels
      ``` python
      anchors = anchors.view(1,3,1,1,2)
      ```
      $$ pred \xrightarrow{\times anchors} pred(pixel)$$

![请添加图片描述](https://i-blog.csdnimg.cn/direct/3a64755eefaa4768950a74392e91cb04.png)
