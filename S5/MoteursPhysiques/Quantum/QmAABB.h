//
// Created by zelie on 10/11/24.
//

#ifndef QMAABB_H
#define QMAABB_H
#include <glm/vec3.hpp>

namespace Quantum {

class QmAABB {
    public:
    glm::vec3 min;
    glm::vec3 max;

    QmAABB(const glm::vec3& min, const glm::vec3& max);

    [[nodiscard]] bool intersect(const QmAABB& other) const;
};

} // Quantum

#endif //QMAABB_H
