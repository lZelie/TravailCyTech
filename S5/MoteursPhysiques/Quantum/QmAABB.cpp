//
// Created by zelie on 10/11/24.
//

#include "QmAABB.h"

namespace Quantum {
    QmAABB::QmAABB(const glm::vec3& min, const glm::vec3& max): min(min),
                                                                max(max)
    {
    }

    bool QmAABB::intersect(const QmAABB& other) const
    {
        return min.x <= other.max.x && max.x >= other.min.x
            && min.y <= other.max.y && max.y >= other.min.y
            && min.z <= other.max.z && max.z >= other.min.z;
    }
} // Quantum