//
// Created by zelie on 10/11/24.
//

#ifndef QMCONTACT_H
#define QMCONTACT_H
#include <mutex>

#include "QmBody.h"

namespace Quantum
{
    class QmContact
    {
    public:
        QmBody& body1;
        QmBody& body2;
        glm::vec3 normal;
        float depth;
        QmContact(QmBody& body1, QmBody& body2, const glm::vec3& normal, float depth);
        QmContact(QmBody& body1, QmBody& body2);
    };
} // Quantum

#endif //QMCONTACT_H
