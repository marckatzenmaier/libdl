//
// Created by marc on 17.05.19.
//

#include "libdl/Optimizer.h"

void SGD_Optimizer::optimize(){
    //std::cout<<"optim1"<<std::endl;
    for(auto& i:variable_vec){
        //std::cout<<i->getName()<<std::endl;
        //std::cout<<"data"<<i->getData()<<std::endl;
        //std::cout<<"grad"<<i->getGradient().dimension(0)<<"grad"<<i->getGradient().dimension(1)<<"grad"<<i->getGradient().dimension(2)<<"grad"<<i->getGradient().dimension(3)<<std::endl;
        i->setData(i->getData() - learning_rate*i->getGradient());
    }

    //std::cout<<"optim2"<<std::endl;
}