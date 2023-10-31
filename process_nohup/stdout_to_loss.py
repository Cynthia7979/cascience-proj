

path = './nohup.out'
epoch_all_path = './output/{}_all'
loss_all_path = './output/loss_all'
acc_all_path = './output/acc_all'

def main():
    f = open(path, 'r')
    loss_f = open(loss_all_path, 'w')
    acc_f = open(acc_all_path, 'w')

    current_epoch = '0'
    epoch_f = open(epoch_all_path.format(current_epoch), 'w')
    for line in f.readlines():
        if '[E: ' not in line:
            if 'tensor(' in line: continue
            else:
                avg_acc = get_content(line, 'avg acc: ', '\n')
                acc_f.write(avg_acc+'\n')
        elif f"[E: {current_epoch}]" not in line:
            avg_loss = get_content(line, ", avg_loss: ", '\n')
            loss_f.write(avg_loss+'\n')
            current_epoch = get_content(line, '[E: ', ']')
            epoch_f.close()
            epoch_f = open(epoch_all_path.format(current_epoch), 'w')
            
        loss = get_content(line, f"[E: {current_epoch}] loss: ", ",")
        epoch_f.write(loss+'\n')
        print(loss)


def get_content(text, before, after):
    return text[text.find(before)+len(before):text.find(after)]


if __name__ == "__main__":
    main()
