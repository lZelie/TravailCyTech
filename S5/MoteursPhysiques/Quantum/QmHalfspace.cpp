//
// Created by zelie on 10/21/24.
//

#include "QmHalfspace.h"

namespace Quantum
{
    QmHalfspace::QmHalfspace(const glm::vec3 position, const glm::vec3 normal, const float offset):
        QmBody(TYPE_HALFSPACE), position(position), normal(normal), offset(offset)
    {
    }

    void QmHalfspace::computeAccelerations(unsigned i)
    {
    }

    void QmHalfspace::integrate(float t)
    {
    }

    void QmHalfspace::integrate(float t, unsigned i)
    {
    }

    void QmHalfspace::integrateRK4(float t)
    {
    }

    void QmHalfspace::addForce(const glm::vec3& force, unsigned i)
    {
    }

    void QmHalfspace::reset()
    {
    }

    QmAABB QmHalfspace::getAABB() const
    {
        constexpr glm::vec3 u{1.0f};
        glm::vec3 v1 = cross(normal, u);
        glm::vec3 v2 = cross(normal, v1);
        v1 = std::numeric_limits<float>::max() * normalize(v1);
        v2 = std::numeric_limits<float>::max() * normalize(v2);
        glm::vec3 a = offset * normal + v1 + v2;
        glm::vec3 b = offset * normal - v1 + v2;
        glm::vec3 c = offset * normal - v1 - v2;
        glm::vec3 d = offset * normal + v1 - v2;

        float min_x = std::min(std::min(a.x, b.x), std::min(c.x, d.x));
        float min_y = std::min(std::min(a.y, b.y), std::min(c.y, d.y));
        float min_z = std::min(std::min(a.z, b.z), std::min(c.z, d.z));
        float max_x = std::max(std::max(a.x, b.x), std::max(c.x, d.x));
        float max_y = std::max(std::max(a.y, b.y), std::max(c.y, d.y));
        float max_z = std::max(std::max(a.z, b.z), std::max(c.z, d.z));

        glm::vec3 min{min_x, min_y, min_z};
        glm::vec3 max{max_x, max_y, max_z};

        if (normal.x > 0)
        {
            min.x = min.x - std::numeric_limits<float>::max() * normal.x;
        }
        else
        {
            max.x = max.x - std::numeric_limits<float>::max() * normal.x;
        }
        if (normal.y > 0)
        {
            min.y = min.y - std::numeric_limits<float>::max() * normal.y;
        }
        else
        {
            max.y = max.y - std::numeric_limits<float>::max() * normal.y;
        }
        if (normal.z > 0)
        {
            min.z = min.z - std::numeric_limits<float>::max() * normal.z;
        }
        else
        {
            max.z = max.z - std::numeric_limits<float>::max() * normal.z;
        }

        return {min, max};
    }

    glm::vec3 QmHalfspace::getPosition() const
    {
        return normal * offset;
    }

    void QmHalfspace::setPosition(const glm::vec3& position)
    {
    }

    glm::vec3 QmHalfspace::getVelocity() const
    {
        return {0.0f, 0.0f, 0.0f};
    }

    void QmHalfspace::setVelocity(const glm::vec3& velocity)
    {
    }

    float QmHalfspace::getRestitution()
    {
        return 0.0f;
    }

    float QmHalfspace::getMass() const
    {
        return MAXFLOAT;
    }

    float QmHalfspace::getRadius() const
    {
        return MAXFLOAT;
    }
} // Quantum
