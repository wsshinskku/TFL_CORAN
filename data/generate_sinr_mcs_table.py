# data/generate_sinr_mcs_table.py
# env에서 사용할 SINR→MCS 매핑 테이블(간이)을 CSV로 생성합니다.
import os
import csv

def main(out_csv="data/tables/sinr_mcs.csv"):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    # MCS index, threshold_dB, tb_bits_per_1ms(100MHz) — ChannelModel와 호환
    rows = [
        # idx, thr_dB, tb_bits
        (0,  -5,  20000),
        (1,  -3,  28000),
        (2,  -1,  38000),
        (3,   1,  50000),
        (4,   3,  66000),
        (5,   5,  86000),
        (6,   7, 110000),
        (7,   9, 140000),
        (8,  11, 180000),
        (9,  13, 230000),
        (10, 15, 290000),
        (11, 17, 360000),
        (12, 19, 450000),
        (13, 21, 560000),
        (14, 23, 700000),
        (15, 25, 880000),
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mcs_index","threshold_db","tb_bits"])
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()
