from tqdm import tqdm

def read(fp: str, n):
    i = 0
    lines = []  # a buffer to cache lines

    with open(fp) as f:
        for line in f:
            i += 1
            lines.append(line.strip())  # append a line

            if i >= n:
                yield lines

                # reset buffer
                i = 0
                lines.clear()

    # remaining lines
    if i > 0:
        yield lines


relation2id = {}
entity2id = {}
id2entity = {}
id2relation = {}
entity_num = 0
relation_num = 0

f1 = open("entity2id.txt", "r")
f2 = open("relation2id.txt", "r")

for line in f1:
    seg = line.strip().split()
    entity2id[seg[0]] = int(seg[1])
    id2entity[int(seg[1])] = seg[0]
    entity_num += 1
f1.close()

for line in f2:
    seg = line.strip().split()
    relation2id[seg[0]] = int(seg[1])
    id2relation[int(seg[1])] = seg[0]
    id2relation[int(seg[1]) + 237] = "-" + seg[0]
    relation_num += 1
f2.close()


def write_path(mode: str = "train"):
    f_kb = "{}_pra3.txt".format(mode)
    g = open("{}s3_w.txt".format(mode), "w")
    k = open("{}p3_w.txt".format(mode), "w")
    g2 = open("{}s2_w.txt".format(mode), "w")
    k2 = open("{}p2_w.txt".format(mode), "w")
    g1 = open("{}s1_w.txt".format(mode), "w")
    k1 = open("{}p1_w.txt".format(mode), "w")

    lines_gen = read(f_kb, 2)
    count = 0
    for line in tqdm(lines_gen):
        seg1 = line[0].strip().split()
        # h t r
        #print(seg1[0] + "\t" + seg1[1] + "\t" + id2relation[int(seg1[2])])
        seg2 = line[1].strip().split()
        rel_path3 = []
        rel_path2 = []
        rel_path1 = []
        pr_path3 = 0
        pr_path2 = 0
        pr_path1 = 0
        for i in range(int(seg2[0])):
            num = int(seg2[1])
            if num == 1:
                if float(seg2[3]) > pr_path1:
                    pr_path1 = float(seg2[3])
                    rel_path1.clear()
                    rel_path1.append(int(seg2[2]))  # add direct relation
                    rel_path1.append(pr_path1)  # Confidence
                seg2.pop(1)
                seg2.pop(1)
                seg2.pop(1)

            if num == 2:
                if float(seg2[4]) > pr_path2:
                    pr_path2 = float(seg2[4])
                    rel_path2.clear()
                    rel_path2.append(int(seg2[2]))  # add r1 r2
                    rel_path2.append(int(seg2[3]))
                    rel_path2.append(pr_path2)  # 分数
                seg2.pop(1)
                seg2.pop(1)
                seg2.pop(1)
                seg2.pop(1)
            if num == 5:
                if float(seg2[7]) > pr_path3:
                    pr_path3 = float(seg2[7])
                    rel_path3.clear()
                    rel_path3.append(int(seg2[2]))  # add r1 e1 r2 e2 r3
                    rel_path3.append(int(seg2[3]))
                    rel_path3.append(int(seg2[4]))
                    rel_path3.append(int(seg2[5]))
                    rel_path3.append(int(seg2[6]))
                    rel_path3.append(pr_path3)  # Confidence
                seg2.pop(1)
                seg2.pop(1)
                seg2.pop(1)
                seg2.pop(1)
                seg2.pop(1)
                seg2.pop(1)
                seg2.pop(1)
        if len(rel_path3) > 0:
            # print(str(rel_path3[0]) + "\t" + str(rel_path3[1]) + "\t" + str(rel_path3[2]) + "\t" + str(
            #     rel_path3[3]) + "\t" + str(
            #     rel_path3[4]) + "\n")
            g.write(seg1[0] + "\t" + seg1[1] + "\t" + id2relation[int(seg1[2])] + "\n")
            k.write(
                seg1[0] + "\t" + str(rel_path3[0]) + "\t" + str(rel_path3[1]) + "\t" + str(rel_path3[2]) + "\t" + str(
                    rel_path3[3])
                + "\t" + str(rel_path3[4]) + "\t" + seg1[1] + "\t" + str(rel_path3[5]) + "\n")

        else:
            pass
        if len(rel_path2) > 0:
           # print(str(rel_path2[0]) + "\t" + str(rel_path2[1]) + "\n")
            g2.write(seg1[0] + "\t" + seg1[1] + "\t" + id2relation[int(seg1[2])] + "\n")
            k2.write(
                seg1[0] + "\t" + str(rel_path2[0]) + "\t" + str(rel_path2[1]) + "\t" + seg1[1] + "\t" + str(
                    rel_path2[2]) + "\n")

        else:
            pass
        if len(rel_path1) > 0:
           # print(str(rel_path1[0]) + "\n")
            g1.write(seg1[0] + "\t" + seg1[1] + "\t" + id2relation[int(seg1[2])] + "\n")
            k1.write(
                seg1[0] + "\t" + str(rel_path1[0]) + "\t" + seg1[1] + "\t" + str(rel_path1[1]) + "\n")

        else:
            pass
    g.close()
    k.close()

    # print(entity_num)
    # print(relation_num)
def write_all_path(mode: str = "all"):
    f_kb = "path3.txt".format(mode)
    #g = open("{}s3_w.txt".format(mode), "w")
    k = open("{}p3_w.txt".format(mode), "w")
    #g2 = open("{}s2_w.txt".format(mode), "w")
    k2 = open("{}p2_w.txt".format(mode), "w")
    #g1 = open("{}s1_w.txt".format(mode), "w")
    k1 = open("{}p1_w.txt".format(mode), "w")

    lines_gen = read(f_kb, 2)
    count = 0
    for line in lines_gen:
        seg1 = line[0].strip().split()
        # h t r
        #print(seg1[0] + "\t" + seg1[1] + "\t" + id2relation[int(seg1[2])])
        seg2 = line[1].strip().split()
        rel_path3 = []
        rel_path2 = []
        rel_path1 = []
        pr_path3 = 0
        pr_path2 = 0
        pr_path1 = 0
        for i in range(int(seg2[0])):
            num = int(seg2[1])
            if num == 1:
                if float(seg2[3]) > pr_path1:
                    pr_path1 = float(seg2[3])
                    rel_path1.clear()
                    rel_path1.append(int(seg2[2]))  # add direct relation
                    rel_path1.append(pr_path1)  # Confidence
                seg2.pop(1)
                seg2.pop(1)
                seg2.pop(1)

            if num == 2:
                if float(seg2[4]) > pr_path2:
                    pr_path2 = float(seg2[4])
                    rel_path2.clear()
                    rel_path2.append(int(seg2[2]))  # add r1 r2
                    rel_path2.append(int(seg2[3]))
                    rel_path2.append(pr_path2)  # 分数
                seg2.pop(1)
                seg2.pop(1)
                seg2.pop(1)
                seg2.pop(1)
            if num == 5:
                if float(seg2[7]) > pr_path3:
                    pr_path3 = float(seg2[7])
                    rel_path3.clear()
                    rel_path3.append(int(seg2[2]))  # add r1 e1 r2 e2 r3
                    rel_path3.append(int(seg2[3]))
                    rel_path3.append(int(seg2[4]))
                    rel_path3.append(int(seg2[5]))
                    rel_path3.append(int(seg2[6]))
                    rel_path3.append(pr_path3)  # Confidence
                seg2.pop(1)
                seg2.pop(1)
                seg2.pop(1)
                seg2.pop(1)
                seg2.pop(1)
                seg2.pop(1)
                seg2.pop(1)
        if len(rel_path3) > 0:
            # print(str(rel_path3[0]) + "\t" + str(rel_path3[1]) + "\t" + str(rel_path3[2]) + "\t" + str(
            #     rel_path3[3]) + "\t" + str(
            #     rel_path3[4]) + "\n")
            #g.write(seg1[0] + "\t" + seg1[1] + "\t" + id2relation[int(seg1[2])] + "\n")
            k.write(
                seg1[0] + "\t" + str(rel_path3[0]) + "\t" + str(rel_path3[1]) + "\t" + str(rel_path3[2]) + "\t" + str(
                    rel_path3[3])
                + "\t" + str(rel_path3[4]) + "\t" + seg1[1] + "\t" + str(rel_path3[5]) + "\n")

        else:
            pass
        if len(rel_path2) > 0:
            #print(str(rel_path2[0]) + "\t" + str(rel_path2[1]) + "\n")
            #g2.write(seg1[0] + "\t" + seg1[1] + "\t" + id2relation[int(seg1[2])] + "\n")
            k2.write(
                seg1[0] + "\t" + str(rel_path2[0]) + "\t" + str(rel_path2[1]) + "\t" + seg1[1] + "\t" + str(
                    rel_path2[2]) + "\n")

        else:
            pass
        if len(rel_path1) > 0:
           # print(str(rel_path1[0]) + "\n")
            #g1.write(seg1[0] + "\t" + seg1[1] + "\t" + id2relation[int(seg1[2])] + "\n")
            k1.write(
                seg1[0] + "\t" + str(rel_path1[0]) + "\t" + seg1[1] + "\t" + str(rel_path1[1]) + "\n")

        else:
            pass
    #g.close()
    k.close()

    # print(entity_num)
    # print(relation_num)
write_all_path("all")
write_path("train")
write_path("test")
write_path("valid")