#include <iostream>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <errno.h>
#include <cstdlib>
#include <sys/time.h>
#include <torch/script.h>
#include <torch/torch.h>
#include "model_data.h"  // 包含生成的头文件

// 获取时间戳函数
long long get_timestamp() {
    long long tmp;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    tmp = tv.tv_sec;
    tmp = tmp * 1000 * 1000;
    tmp = tmp + tv.tv_usec;
    return tmp;
}

// 设置串口参数
int set_interface_attribs(int fd, int speed) {
    struct termios tty;
    memset(&tty, 0, sizeof tty);
    if (tcgetattr(fd, &tty) != 0) {
        perror("tcgetattr");
        return -1;
    }

    cfsetispeed(&tty, speed);
    cfsetospeed(&tty, speed);

    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~PARENB;
    tty.c_c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_oflag &= ~OPOST;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cc[VMIN] = 1;
    tty.c_cc[VTIME] = 0;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        return -1;
    }

    return 0;
}

int main(int argc, const char* argv[]) {
    // 加载模型
    torch::jit::script::Module module;
    try {
        // 使用模型数据加载模块
        module = torch::jit::load(std::string((char*)model_data, model_data_len));
    }
    catch (const c10::Error &e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    // 打开串口
    std::string port = "/dev/ttyS7";
    int fd = open(port.c_str(), O_RDWR | O_NOCTTY);
    if (fd == -1) {
        perror("open_port: Unable to open port");
        return -1;
    }
    if (set_interface_attribs(fd, B115200) == -1) {
        close(fd);
        return -1;
    }

    sleep(1);

    // 获取模型输入张量
    std::vector<float> input_data(1250, 0); // 假设输入大小为1250
    at::Tensor input_tensor = torch::from_blob(input_data.data(), {1, 1, 1250, 1}, torch::kFloat);

    uint8_t result_new[8] = {0x55, 0xaa, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    union r_dat {
        float fdat;
        uint8_t udat[4];
    };
    struct r_tm {
        float *cost;
    };
    volatile struct r_tm rtm;
    rtm.cost = (float *)&result_new[4];

    while (true) {
        uint8_t dat;
        int len = 0, index = 0, datalen = 0;
        int status = 0;
        bool flag = false;
        int ret;

        while (true) {
            if ((ret = read(fd, &dat, 1)) < 0) {
                len = 0;
                index = 0;
                datalen = 0;
                status = 0;
            }

            switch (status) {
                case 0:
                    if (dat == 0xaa) {
                        status = 1;
                    }
                    break;
                case 1:
                    if (dat == 0x55) {
                        status = 2;
                        index = 0;
                    } else {
                        status = 0;
                    }
                    break;
                case 2:
                    if (index == 0) {
                        datalen = dat;
                        index = 1;
                    } else {
                        datalen |= dat << 8;
                        status = 3;
                        len = 0;
                        index = 0;
                    }
                    break;
                case 3:
                    r_dat rdat;
                    rdat.udat[index++] = dat;
                    if (index >= 4) {
                        index = 0;
                        input_data[len++] = rdat.fdat;
                    }
                    if (len >= datalen) {
                        flag = true;
                        status = 0;
                    } else if (len >= 1250) {
                        status = 0;
                    }
                    break;
                default:
                    status = 0;
                    break;
            }
            if (flag) {
                break;
            }
        }

        if (flag) {
            long long start_time = get_timestamp();
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            at::Tensor output = module.forward(inputs).toTensor();
            long long end_time = get_timestamp();
            long long cost_time = end_time - start_time;

            *rtm.cost = (float)(cost_time / 1000.0);
            result_new[2] = 0x00;

            float result = output.item<float>();
            result_new[3] = result > 0.5 ? 0x01 : 0x00;

            write(fd, result_new, 8);
        } else if (ret < 0) {
            result_new[2] = 0x01;
            write(fd, result_new, 4);
        }
    }

    close(fd);
    return 0;
}
