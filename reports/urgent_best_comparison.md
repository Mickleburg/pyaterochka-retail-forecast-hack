# Срочное сравнение best-like сабмитов

База A: `submissions/test_cluster_temporal_blend_v2.csv`, подтвержденный score `95.93`.

## Отличия файлов от A

```text
              name                                        filename  mean_abs_delta_vs_A  mean_rel_delta_vs_A  mean_abs_rel_delta_vs_A  max_rel_delta_vs_A  share_delta_gt_0p1pct  share_delta_gt_0p3pct  share_delta_gt_1pct  corr_with_A  share_changed_stores
A_cluster_temporal  submissions/test_cluster_temporal_blend_v2.csv             0.000000             0.000000                 0.000000            0.000000               0.000000               0.000000             0.000000     1.000000              0.000000
          B_decile submissions/test_decile_cluster_temporal_v1.csv         45552.660835             0.000975                 0.001350            0.010766               0.474703               0.108804             0.000485     0.999993              1.000000
           C_huber          submissions/test_temporal_huber_v2.csv         72930.678453             0.001678                 0.002276            0.030041               0.593888               0.344555             0.006500     0.999985              1.000000
        D_logratio submissions/test_temporal_logratio_blend_v2.csv         72119.938050             0.001822                 0.002250            0.016005               0.492991               0.355130             0.008732     0.999985              1.000000
        E_prevbest           submissions/test_cluster_blend_v1.csv         86028.598358             0.001996                 0.002686            0.022074               0.627019               0.373126             0.018967     0.999978              1.000000
    F_old_temporal    submissions/test_temporal_ridge_ratio_v1.csv         77772.719889             0.001352                 0.002381            0.039353               0.659908               0.315741             0.010914     0.999980              1.000000
        G_ridge_v2          submissions/test_temporal_ridge_v2.csv         78494.177733             0.001572                 0.002433            0.024669               0.674412               0.359010             0.008829     0.999983              0.998545
```

## Где decile_cluster_temporal сильнее отличается от A

```text
     segment_type segment_value    n  mean_abs_rel_B_vs_A  max_abs_rel_B_vs_A
     trend_bucket             0 1754             0.002591            0.010766
           regime  october_drop 3351             0.001904            0.010766
volatility_bucket             4 4123             0.001647            0.010766
       rto_decile             8 2061             0.001611            0.010694
           regime     declining  136             0.001598            0.003728
           regime   high_stable 1129             0.001562            0.009796
volatility_bucket             0 4123             0.001561            0.009796
     trend_bucket             1 1574             0.001515            0.009221
       rto_decile             6 2061             0.001454            0.009055
           regime       growing 3897             0.001450            0.007777
       rto_decile             5 2061             0.001399            0.009410
     trend_bucket             3 5160             0.001368            0.009796
       rto_decile             7 2062             0.001361            0.010766
     trend_bucket             2 2636             0.001353            0.008621
volatility_bucket             1 4123             0.001345            0.007792
       rto_decile             3 2061             0.001327            0.009430
       rto_decile             9 2062             0.001323            0.010560
       rto_decile             4 2062             0.001322            0.008829
           regime        stable 4059             0.001317            0.007025
       rto_decile             0 2062             0.001257            0.010271
       rto_decile             2 2062             0.001235            0.008265
       rto_decile             1 2061             0.001211            0.008285
           regime  low_volatile 2273             0.001197            0.010271
volatility_bucket             2 4123             0.001120            0.010694
           regime      volatile 2736             0.001108            0.009086
     trend_bucket             4 9491             0.001083            0.008614
volatility_bucket             3 4123             0.001076            0.010441
           regime october_spike 3034             0.000895            0.009657
```

## Вывод

- `decile_cluster_temporal_v1` близок к A и уже подтвердил `95.92`, поэтому A/B blends и A/B selector имеют самый высокий приоритет.
- Чистый temporal_ridge_v2 получил `95.90`, поэтому temporal нужно применять только как небольшой blend или override на выбранных сегментах.
- Blends с previous best нужны как safety-контроль, но не являются главным направлением.
