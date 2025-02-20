//
// Created by cytech on 27/09/24.
//

#ifndef APPLICATION_QMDRAG_H
#define APPLICATION_QMDRAG_H

#include "QmForceGenerator.h"

namespace Quantum {
    constexpr int DRAG = 2;
    class QmDrag final : public QmForceGenerator {
        float k1 = 0;
        float k2 = 0;

    public:
        QmDrag(float k1, float k2);
        void update(QmParticle *particle, unsigned int i) override;
    };

} // Quantum

#endif //APPLICATION_QMDRAG_H
