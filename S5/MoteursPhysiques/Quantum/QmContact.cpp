//
// Created by zelie on 10/11/24.
//

#include "QmContact.h"

namespace Quantum
{
    QmContact::QmContact(QmBody& body1, QmBody& body2, const glm::vec3& normal, const float depth):
        body1(body1),
        body2(body2),
        normal(normal),
        depth(depth)
    {
    }

    QmContact::QmContact(QmBody& body1, QmBody& body2):
        body1(body1),
        body2(body2),
        normal(0),
        depth(0)
    {
    }
} // Quantum
