#ifndef QMPARTICLE_H
#define QMPARTICLE_H

#include <array>
#include <Application/GxParticle.h>
#include <Application/GxParticle.h>
#include <Application/GxParticle.h>
#include <Application/GxParticle.h>
#include <Application/GxParticle.h>
#include <Application/GxParticle.h>
#include <Application/GxParticle.h>
#include <Application/GxParticle.h>
#include <glm/glm.hpp>

#include "QmAABB.h"
#include "QmBody.h"


namespace Quantum {
	class QmUpdater;

	class QmParticle : public QmBody {
	public:
		QmParticle();
		QmParticle(glm::vec3 position, glm::vec3 velocity, glm::vec3 acceleration, float mass, int charge, float radius);
		QmParticle(glm::vec3 position, glm::vec3 velocity, glm::vec3 acceleration, float mass, int charge, float radius, float damping);
		~QmParticle() override;
		void integrate(float t) override;
		void integrate(float, unsigned int i) final;
		void computeAccelerations(unsigned i) override;
		void integrateRK4(float t) override;
		void addForce(const glm::vec3 &force, unsigned int i) override;

        void reset() override;

		[[nodiscard]] std::array<glm::vec3, 4> getAcc() const;
		[[nodiscard]] std::array<glm::vec3, 4> getVel() const;
		[[nodiscard]] std::array<glm::vec3, 4> getPos() const;

        [[nodiscard]] int getCharge() const;

        void setUpdater(QmUpdater* updater);
		[[nodiscard]] QmUpdater* getUpdater() const;

		void setDamping(float damping);
		[[nodiscard]] float getDamping() const;

		[[nodiscard]] QmAABB getAABB() const final;

		[[nodiscard]] glm::vec3 getPosition() const final;
		[[nodiscard]] float getRadius() const final;
		void setPosition(const glm::vec3& position) final;
		[[nodiscard]] glm::vec3 getVelocity() const final;
		void setVelocity(const glm::vec3& velocity) final;
		[[nodiscard]] float getMass() const override;
		[[nodiscard]] float getRestitution() override;

	private:
		QmUpdater* updater = nullptr;
		std::array<glm::vec3, 4> position;
		std::array<glm::vec3, 4> velocity;
		std::array<glm::vec3, 4> acceleration;
        std::array<glm::vec3, 4> forceAccumulator{};
        int charge = 1;
        float invMass{};

		float damping{};
		float radius;
		float restitution{1};

	};
}

#endif