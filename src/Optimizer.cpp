//
// Created by marc on 17.05.19.
//

#include "libdl/Optimizer.h"

void SGD_Optimizer::optimize(){
    for(auto& i:variable_vec){
        i->setData(i->getData() - learning_rate*i->getGradient());
    }
}