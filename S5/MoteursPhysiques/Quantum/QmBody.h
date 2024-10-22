#pragma once
#ifndef QMBODY_H
#define QMBODY_H

#include <mutex>
#include <Application/GxParticle.h>
#include <Application/GxParticle.h>
#include <glm/glm.hpp>

#include "QmAABB.h"
#include "QmUpdater.h"

namespace Quantum
{
    constexpr int TYPE_PARTICLE = 0;

    class QmBody
    {
    public:
        virtual void computeAccelerations(unsigned int i) = 0;
        virtual void integrate(float t) = 0;
        virtual void integrate(float t, unsigned int i) = 0;
        virtual void integrateRK4(float t) = 0;


        virtual void addForce(const glm::vec3& force, unsigned int i) = 0;

        virtual void reset() = 0;

        [[nodiscard]] int getType() const { return type; }
        [[nodiscard]] virtual QmAABB getAABB() const = 0;
        [[nodiscard]] virtual glm::vec3 getPosition() const = 0;
        virtual void setPosition(const glm::vec3& position) = 0;
        [[nodiscard]] virtual glm::vec3 getVelocity() const = 0;
        virtual void setVelocity(const glm::vec3& velocity) = 0;
        [[nodiscard]] virtual float getRestitution() = 0;
        [[nodiscard]] virtual float getMass() const = 0;
        [[nodiscard]] virtual float getRadius() const = 0;
        virtual ~QmBody() = default;
        mutable  std::mutex mutex{};

    protected:
        explicit QmBody(int type) : type(type)
        {
        }

    private:
        int type = -1;
    };
}

#endif
