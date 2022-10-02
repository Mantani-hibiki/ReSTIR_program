
#pragma once

#ifndef NS_RENDERER_ReSTIR_light_TRACER_HPP
#define NS_RENDERER_ReSTIR_light_TRACER_HPP

#include"../base.hpp"
#include"../../parallel.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace ns {

	///////////////////////////////////////////////////////////////////////////////////////////////////

	namespace rndr {

		///////////////////////////////////////////////////////////////////////////////////////////////////

		namespace ReSTIR_light {

			///////////////////////////////////////////////////////////////////////////////////////////////////
			//renderer
			///////////////////////////////////////////////////////////////////////////////////////////////////

			class reservoir;

			class renderer : public rndr::renderer
			{
			public:

				//�R���X�g���N�^
				renderer(const scene& scene, const camera& camera, const size_t spp, const size_t nt = std::thread::hardware_concurrency());

				//�����_�����O
				virtual imagef render(const scene& scene, const camera& camera) const;

			private:

				//WRS��p����RIS�v�Z
				reservoir WRS_RIS(const scene& scene, const intersection& isect, const bsdf& bsdf, const wavelength& lambda, area_lights const& area_lights, random_number_generator& rng) const;

				//reservoir�̌���
				reservoir combine_reservoirs(const scene& scene, reservoir r[], int r_size, random_number_generator& rng, const wavelength& lambda, memory_allocator& allocator, int check, int biase_check) const;

				//�s�N�Z�����Ƃ�reservoir�����߂�
				reservoir RIS_pixel(const scene& scene, const wavelength& lambda, ray r, random_number_generator& rng, memory_allocator& allocator) const;

				//�s�N�Z�����Ƃ̋P�x�����߂�
				spectrum radiance_pixel(reservoir& reservoir, const scene& scene, const wavelength& lambda, memory_allocator& allocator) const;

			private:

				size_t m_spp;
				size_t m_nt;
			};

			//RIS�d�݂ƃT���v���������reservoir�N���X
			class reservoir {
				point_sample y;
				float wsum = 0;
				float W = 0;
				float M = 0;
				intersection isect;
				direction dir;
			public:
				//�T���v����I�тȂ���C�d�݂̍��v�l�E�T���v�������X�V
				void update(point_sample x, float w, float random) {
					wsum += w;
					M += 1;
					if (wsum != 0) {
						if (random < (w / wsum)) {
							y = x;
						}
					}
				};

				point_sample get_sample() { return y; };
				void set_sample(point_sample new_sample) { y = new_sample; };

				float get_wsum() { return wsum; };
				void set_wsum(float new_wsum) { wsum = new_wsum; };

				float get_W() { return W; };
				void set_W(float new_W) { W = new_W; };

				float get_M() { return M; };
				void set_M(float new_M) { M = new_M; };

				intersection get_isect() { return isect; };
				void set_isect(intersection new_isect) { isect = new_isect; };

				direction get_dir() { return dir; };
				void set_dir(direction new_dir) { dir = new_dir; };
			};

			///////////////////////////////////////////////////////////////////////////////////////////////////

		} //namespace ReSTIR_light

		///////////////////////////////////////////////////////////////////////////////////////////////////

	} //namespace rndr

	///////////////////////////////////////////////////////////////////////////////////////////////////

} //namespace ns

///////////////////////////////////////////////////////////////////////////////////////////////////

#include"ReSTIR_light/ReSTIR_light-impl.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////

#endif
