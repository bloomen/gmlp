#include <fstream>
#include <sstream>

#include "Network.h"
#include "utils.h"

int main()
{
//    std::vector<std::vector<float>> X = {
//        {2.7810836,2.550537003},
//        {1.465489372,2.362125076},
//        {3.396561688,4.400293529},
//        {1.38807019,1.850220317},
//        {3.06407232,3.005305973},
//        {7.627531214,2.759262235},
//        {5.332441248,2.088626775},
//        {6.922596716,1.77106367},
//        {8.675418651,-0.242068655},
//        {7.673756466,3.508563011},
//    };
//    std::vector<std::vector<float>> y = {
//        {1, 0},
//        {1, 0},
//        {1, 0},
//        {1, 0},
//        {1, 0},
//        {0, 1},
//        {0, 1},
//        {0, 1},
//        {0, 1},
//        {0, 1},
//    };

    std::ifstream f{"/home/cblume/workspace/cbnn/data/boston.csv"};
    std::vector<std::vector<float>> X;
    std::vector<std::vector<float>> y;
    while (!f.eof())
    {
        X.emplace_back(13);
        y.emplace_back(1);
        for (std::size_t i = 0; i < 13; ++i)
        {
            f >> X.back()[i];
        }
        f >> y.back()[0];
    }

    gmlp::init::DefaultRandomEngine engine{42};

    // Ref with 100 epochs
    // Class: MAE=0.1428
    // Regre: MAE=2.7155
    const auto target_type = gmlp::Regression;
    gmlp::Network net{target_type, {13, 13, 1}, engine};
    net.print();

    float learning_rate = 0.01f;
    if (target_type == gmlp::Classification)
    {
        learning_rate = 0.5f;
        for (auto& value : y)
        {
            if (value[0] > 22)
            {
                value[0] = 1;
            }
            else
            {
                value[0] = 0;
            }
        }
    }

    const auto split = gmlp::split_train_test(X, y, 0.3f, engine);

    std::cout << "training with " << split.X_train.size() << " samples" << std::endl;
    for (std::size_t i = 0; i < 100; ++i)
    {
        const auto loss = net.train(split.X_train, split.y_train, learning_rate);
        std::cout << "epoch=" << i << " " << "loss=" << loss << std::endl;
    }

    std::stringstream ss;
    net.save(ss);
    ss.seekg(0);
    auto loaded_net = gmlp::Network::load(ss);

    std::cout << "prediction" << std::endl;
    std::vector<std::vector<float>> pred;
    for (const auto& row : split.X_test)
    {
        pred.push_back(loaded_net.predict(row));
    }

    if (target_type == gmlp::Classification)
    {
        for (auto& value : pred)
        {
            value[0] = value[0] > 0.5f ? 1.0f : 0.0f;
        }
    }

    for (std::size_t i = 0; i < 10; ++i)
    {
        std::cout << "truth=" << split.y_test[i][0] << " " << "pred=" << pred[i][0] << std::endl;
    }

    std::cout << "MAE=" << gmlp::mae(split.y_test, pred) << std::endl;
}
