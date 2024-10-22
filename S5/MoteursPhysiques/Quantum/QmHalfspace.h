//
// Created by zelie on 10/21/24.
//

#ifndef QM_HALFSPACE_H
#define QM_HALFSPACE_H
#include "QmBody.h"

namespace Quantum
{
    constexpr int TYPE_HALFSPACE = 1;
    class QmHalfspace : public QmBody
    {
    public:
        glm::vec3 position;
        glm::vec3 normal;
        float offset;
        QmHalfspace(glm::vec3 position, glm::vec3 normal, float offset);

        void computeAccelerations(unsigned i) override;
        void integrate(float t) override;
        void integrate(float t, unsigned i) override;
        void integrateRK4(float t) override;
        void addForce(const glm::vec3& force, unsigned i) override;
        void reset() override;
        [[nodiscard]] QmAABB getAABB() const override;
        [[nodiscard]] glm::vec3 getPosition() const override;
        void setPosition(const glm::vec3& position) override;
        [[nodiscard]] glm::vec3 getVelocity() const override;
        void setVelocity(const glm::vec3& velocity) override;
        [[nodiscard]] float getRestitution() override;
        [[nodiscard]] float getMass() const override;
        [[nodiscard]] float getRadius() const override;
    };
} // Quantum

#endif //QM_HALFSPACE_H
