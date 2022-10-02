
#include"../../util/spectrum.hpp"
#define SAMPLE_SIZE 8
#define radius 30
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace ns {

	///////////////////////////////////////////////////////////////////////////////////////////////////

	namespace rndr {

		///////////////////////////////////////////////////////////////////////////////////////////////////

		namespace ReSTIR_light {

			///////////////////////////////////////////////////////////////////////////////////////////////////
			//�O���錾
			///////////////////////////////////////////////////////////////////////////////////////////////////

			using util::accumulation_buffer;

			///////////////////////////////////////////////////////////////////////////////////////////////////
			//bssrdf_to_bsdf_adapter
			///////////////////////////////////////////////////////////////////////////////////////////////////

			class bssrdf_to_bsdf_adapter
			{
			public:

				bssrdf_to_bsdf_adapter(const intersection& isect, const bssrdf& bssrdf, const wavelength& lambda, random_number_generator& rng, memory_allocator& allocator) : mp_isect(&isect), mp_bssrdf(&bssrdf), mp_lambda(&lambda), mp_rng(&rng), mp_allocator(&allocator)
				{
				}
				spectrum f(const direction& wi) const
				{
					if (const auto bsdf = mp_isect->bsdf(wi, *mp_lambda, *mp_allocator)) {
						if (const auto sample = bsdf.sample(*mp_rng, bsdf::type::all_transmission)) {
							return sample.f() * (sample.w().abs_cos() / sample.pdf());
						}
					}
					return spectrum(*mp_lambda);
				}
				float pdf(const direction& wi) const
				{
					return mp_bssrdf->pdf_w(*mp_isect, wi);
				}
				bsdf::type::flags r_type() const
				{
					return bsdf::type::diffuse_reflection;
				}
				bsdf::type::flags t_type() const
				{
					return bsdf::type::flags();
				}

			private:

				const intersection* mp_isect;
				const bssrdf* mp_bssrdf;
				const wavelength* mp_lambda;
				random_number_generator* mp_rng;
				memory_allocator* mp_allocator;
			};

			///////////////////////////////////////////////////////////////////////////////////////////////////
			//renderer
			///////////////////////////////////////////////////////////////////////////////////////////////////

			//�R���X�g���N�^
			inline renderer::renderer(const scene&, const camera&, const size_t spp, const size_t nt)
			{
				m_spp = spp;
				m_nt = nt;
			}

			///////////////////////////////////////////////////////////////////////////////////////////////////

			//�����_�����O
			inline imagef renderer::render(const scene& scene, const camera& camera) const
			{
				//�X�N���[���̏�����
				const int w = camera.res_x();
				const int h = camera.res_y();
				imagef screen(w, h, 3);
				int biase_check = 1;

				//biase����
				if (biase_check == 1) {
					//�ߋ��t���[����reservoir��ۑ�����ϐ�
					static reservoir reservoir_p[512][512];

					//ris���s��reservoir��錾
					reservoir** reservoir_ris = new reservoir * [w];

					//temporal reuse���s��reservoir��錾
					reservoir** reservoir_t = new reservoir * [w];

					reservoir** reservoir_t2 = new reservoir * [w];

					for (int i = 0; i < w; i++) {
						reservoir_ris[i] = new reservoir[h];

						reservoir_t[i] = new reservoir[h];

						reservoir_t2[i] = new reservoir[h];
					}

					//�s�N�Z�����Ƃ�reservoir�����߂�
					in_parallel(std::make_tuple(w, h), [&, this](const std::tuple<int, int>& idx)
						{
							const int x = std::get<0>(idx);
							const int y = std::get<1>(idx);

							thread_local pseudo_rng rng(
								std::random_device{}()
							);
							thread_local memory_allocator allocator;

							//RIS�v�Z
							reservoir_ris[x][y] = RIS_pixel(scene, wavelength(rng), camera.sample(x, y, rng), rng, allocator);

							//�ߋ��t���[����reservoir�ƌ���
							reservoir r[2];
							r[0] = reservoir_ris[x][y];
							r[1] = reservoir_p[x][y];
							
							reservoir_t[x][y] = combine_reservoirs(scene, r, 2, rng, wavelength(rng), allocator, 1, biase_check);
						}, m_nt);

					//spatial reuse��reservoir�̌���
					in_parallel(std::make_tuple(w, h, 2), [&, this](const std::tuple<int, int, int>& idx)
						{
							const int x = std::get<0>(idx);
							const int y = std::get<1>(idx);
							const int z = std::get<2>(idx);

							thread_local pseudo_rng rng(
								std::random_device{}()
							);
							thread_local memory_allocator allocator;

							if (z == 0) {
								//�ߗ׃��U�[�o�̎��W
								reservoir r[6];
								int r_size = sizeof(r) / sizeof(reservoir);
								r[0] = reservoir_t[x][y];

								if (r[0].get_isect()) {
									const auto ng1 = r[0].get_isect().ng();
									int count = 1;

									//�@���x�N�g�����m�̊p�x�����ȉ��̏ꍇ�̂݁C�������郊�U�[�o�Ƃ���
									for (int i = 0; i < 50; i++) {
										int random_x = rng.generate_uniform_int(-radius, radius);
										int random_y = rng.generate_uniform_int(-int(sqrt(radius * radius - random_x * random_x)), int(sqrt(radius * radius - random_x * random_x)));
										if ((((x + random_x) < w) && (x + random_x) >= 0) && (((y + random_y) < h) && (y + random_y) >= 0)) {
											const auto ng2 = reservoir_t[x + random_x][y + random_y].get_isect().ng();
											const auto cos = (ng1.x * ng2.x + ng1.y * ng2.y + ng1.z * ng2.z) / (sqrt(ng1.x * ng1.x + ng1.y * ng1.y + ng1.z * ng1.z) * sqrt(ng2.x * ng2.x + ng2.y * ng2.y + ng2.z * ng2.z));
											if (cos > 0.9063) {
												r[count] = reservoir_t[x + random_x][y + random_y];
												count++;
											}
										}
										if (count == 6) {
											break;
										}
									}
									reservoir_t2[x][y] = combine_reservoirs(scene, r, r_size, rng, wavelength(rng), allocator, 2, biase_check);
								}
								else {
									reservoir_t2[x][y] = reservoir_t[x][y];
								}

							}
							else {
								//�Q��ڂ̋ߗ׃��U�[�o�̎��W
								reservoir r[6];
								int r_size = sizeof(r) / sizeof(reservoir);
								r[0] = reservoir_t2[x][y];

								if (r[0].get_isect()) {
									const auto ng1 = r[0].get_isect().ng();
									int count = 1;

									//�@���x�N�g�����m�̊p�x�����ȉ��̏ꍇ�̂݁C�������郊�U�[�o�Ƃ���
									for (int i = 0; i < 50; i++) {
										int random_x = rng.generate_uniform_int(-radius, radius);
										int random_y = rng.generate_uniform_int(-int(sqrt(radius * radius - random_x * random_x)), int(sqrt(radius * radius - random_x * random_x)));
										if ((((x + random_x) < w) && (x + random_x) >= 0) && (((y + random_y) < h) && (y + random_y) >= 0)) {
											const auto ng2 = reservoir_t2[x + random_x][y + random_y].get_isect().ng();
											const auto cos = (ng1.x * ng2.x + ng1.y * ng2.y + ng1.z * ng2.z) / (sqrt(ng1.x * ng1.x + ng1.y * ng1.y + ng1.z * ng1.z) * sqrt(ng2.x * ng2.x + ng2.y * ng2.y + ng2.z * ng2.z));
											if (cos > 0.9063) {
												r[count] = reservoir_t2[x + random_x][y + random_y];
												count++;
											}
										}
										if (count == 6) {
											break;
										}
									}
									reservoir_p[x][y] = combine_reservoirs(scene, r, r_size, rng, wavelength(rng), allocator, 2, biase_check);
								}
								else {
									reservoir_p[x][y] = reservoir_t2[x][y];
								}

								//�摜�̋P�x�����߂�
								const auto wi = wavelength(rng);
								const spectrum spectrum_ris(
									radiance_pixel(reservoir_ris[x][y], scene, wi, allocator)
								);
								const spectrum spectrum_t(
									radiance_pixel(reservoir_t[x][y], scene, wi, allocator)
								);
								const spectrum spectrum_t2(
									radiance_pixel(reservoir_t2[x][y], scene, wi, allocator)
								);
								const spectrum spectrum_p(
									radiance_pixel(reservoir_p[x][y], scene, wi, allocator)
								);

								accumulation_buffer sum;

								//�d�݂̊����ɉ����āC�P�x�̉��d���ς��Ƃ�
								float Msum = reservoir_ris[x][y].get_M() + reservoir_t[x][y].get_M() + reservoir_t2[x][y].get_M() + reservoir_p[x][y].get_M();

								if (Msum != 0) {
									if (spectrum_ris.is_finite()) {
										sum += spectrum_ris * reservoir_ris[x][y].get_M() / Msum;
									}
									
									if (spectrum_t.is_finite()) {
										sum += spectrum_t * reservoir_t[x][y].get_M() / Msum;
									}
									
									if (spectrum_t2.is_finite()) {
										sum += spectrum_t2 * reservoir_t2[x][y].get_M() / Msum;
									}
									
									if (spectrum_p.is_finite()) {
										sum += spectrum_p * reservoir_p[x][y].get_M() / Msum;
									}
								}
								else {
									if (spectrum_p.is_finite()) {
										sum += spectrum_p;
									}
								}

								const auto avg = rgb_spectrum(sum);
								screen(x, y)[0] = avg[0];
								screen(x, y)[1] = avg[1];
								screen(x, y)[2] = avg[2];

							}
						}, m_nt);

					for (int i = 0; i < w; i++) {
						//ris��reservoir�̃����������
						delete[] reservoir_ris[i];

						//temporal reservoir�̃����������
						delete[] reservoir_t[i];

						delete[] reservoir_t2[i];
					}

					delete[] reservoir_ris;

					delete[] reservoir_t;

					delete[] reservoir_t2;

				}
				//unbiase�̌���
				else {
					//�ߋ��t���[����reservoir��ۑ�����ϐ�
					static reservoir reservoir_p[512][512];

					//ris���s��reservoir��錾
					reservoir** reservoir_ris = new reservoir * [w];

					//temporal reuse���s��reservoir��錾
					reservoir** reservoir_t = new reservoir * [w];

					for (int i = 0; i < w; i++) {
						reservoir_ris[i] = new reservoir[h];

						reservoir_t[i] = new reservoir[h];
					}

					//�s�N�Z�����Ƃ�reservoir�����߂�
					in_parallel(std::make_tuple(w, h), [&, this](const std::tuple<int, int>& idx)
						{
							const int x = std::get<0>(idx);
							const int y = std::get<1>(idx);

							thread_local pseudo_rng rng(
								std::random_device{}()
							);
							thread_local memory_allocator allocator;

							//RIS�v�Z
							reservoir_ris[x][y] = RIS_pixel(scene, wavelength(rng), camera.sample(x, y, rng), rng, allocator);

							//�ߋ��t���[����reservoir�ƌ���
							reservoir r[2];
							r[0] = reservoir_ris[x][y];
							r[1] = reservoir_p[x][y];

							reservoir_t[x][y] = combine_reservoirs(scene, r, 2, rng, wavelength(rng), allocator, 1, 1);

						}, m_nt);

					//spatial reuse��reservoir�̌���
					in_parallel(std::make_tuple(w, h), [&, this](const std::tuple<int, int>& idx)
						{
							const int x = std::get<0>(idx);
							const int y = std::get<1>(idx);

							thread_local pseudo_rng rng(
								std::random_device{}()
							);
							thread_local memory_allocator allocator;

							//�ߗ׃��U�[�o�̎��W
							reservoir r[4];
							int r_size = sizeof(r) / sizeof(reservoir);
							r[0] = reservoir_t[x][y];

							if (r[0].get_isect()) {
								const auto ng1 = r[0].get_isect().ng();
								int count = 1;

								//�@���x�N�g�����m�̊p�x�����ȉ��̏ꍇ�̂݁C�������郊�U�[�o�Ƃ���
								for (int i = 0; i < 50; i++) {
									int random_x = rng.generate_uniform_int(-radius, radius);
									int random_y = rng.generate_uniform_int(-int(sqrt(radius * radius - random_x * random_x)), int(sqrt(radius * radius - random_x * random_x)));
									if ((((x + random_x) < w) && (x + random_x) >= 0) && (((y + random_y) < h) && (y + random_y) >= 0)) {
										const auto ng2 = reservoir_t[x + random_x][y + random_y].get_isect().ng();
										const auto cos = (ng1.x * ng2.x + ng1.y * ng2.y + ng1.z * ng2.z) / (sqrt(ng1.x * ng1.x + ng1.y * ng1.y + ng1.z * ng1.z) * sqrt(ng2.x * ng2.x + ng2.y * ng2.y + ng2.z * ng2.z));
										if (cos > 0.9063) {
											r[count] = reservoir_t[x + random_x][y + random_y];
											count++;
										}
									}
									if (count == 4) {
										break;
									}
								}
								reservoir_p[x][y] = combine_reservoirs(scene, r, r_size, rng, wavelength(rng), allocator, 2, biase_check);
							}
							else {
								reservoir_p[x][y] = reservoir_t[x][y];
							}

							const auto wi = wavelength(rng);
							
							const spectrum spectrum_ris(
								radiance_pixel(reservoir_ris[x][y], scene, wi, allocator)
							);
							const spectrum spectrum_t(
								radiance_pixel(reservoir_t[x][y], scene, wi, allocator)
							);
							const spectrum spectrum_p(
								radiance_pixel(reservoir_p[x][y], scene, wi, allocator)
							);

							accumulation_buffer sum;

							//�d�݂̊����ɉ����āC�P�x�̉��d���ς��Ƃ�
							float Msum = reservoir_ris[x][y].get_M() + reservoir_t[x][y].get_M() + reservoir_p[x][y].get_M();

							if (Msum != 0) {
								if (spectrum_ris.is_finite()) {
									sum += spectrum_ris * reservoir_ris[x][y].get_M() / Msum;
								}
								if (spectrum_t.is_finite()) {
									sum += spectrum_t * reservoir_t[x][y].get_M() / Msum;
								}
								if (spectrum_p.is_finite()) {
									sum += spectrum_p * reservoir_p[x][y].get_M() / Msum;
								}
							}
							else {
								if (spectrum_p.is_finite()) {
									sum += spectrum_p;
								}
							}
							const auto avg = rgb_spectrum(sum);
							screen(x, y)[0] = avg[0];
							screen(x, y)[1] = avg[1];
							screen(x, y)[2] = avg[2];
						}, m_nt);

					for (int i = 0; i < w; i++) {
						//ris��reservoir�̃����������
						delete[] reservoir_ris[i];

						//temporal reservoir�̃����������
						delete[] reservoir_t[i];
					}

					delete[] reservoir_ris;

					delete[] reservoir_t;
				}

				return screen;
			}

			///////////////////////////////////////////////////////////////////////////////////////////////////

			//WRS��p����RIS�v�Z
			reservoir renderer::WRS_RIS(const scene& scene, const intersection& isect, const bsdf& bsdf, const wavelength& lambda, area_lights const& area_lights, random_number_generator& rng) const {
				float random = rng.generate_uniform_real(0.f);
				reservoir reservoir;
				
				//�����̃T���v������pdf�����߂āCreservoir�̍X�V���s��
				for (int i = 0; i < SAMPLE_SIZE; i++) {
					point_sample x = area_lights.sample(rng);
					const vec4 tmp_ld = x.p() - isect.p();
					const float dist2 = squared_norm(tmp_ld);
					const float dist = sqrtf(dist2);
					const auto dir = direction(tmp_ld / dist, isect.ng(), isect.ns());
					const direction lwo(-dir, x.ng(), x.ns());
					auto L = x.Le(lwo, lambda) * bsdf.f(dir) * dir.abs_cos() * lwo.abs_cos();
					float pdf = luminance(L);
					float w = 0;
					if (x.pdf() != 0) {
						w = pdf / x.pdf();
					}
					reservoir.update(x, w, random);
				}

				//�T���v���������Ȓl�Ȃ�W��0�ɂ���
				const auto y = reservoir.get_sample();
				if (y.is_invalid()) {
					return reservoir;
				}

				//���˕����̏����v�Z
				const vec4 tmp_ld = y.p() - isect.p();
				const float dist2 = squared_norm(tmp_ld);
				const float dist = sqrtf(dist2);
				const auto dir = direction(tmp_ld / dist, isect.ng(), isect.ns());
				if (dir.in_darkspot()) {
					return reservoir;
				}
				else if (dir.in_upper_hemisphere()) {
					if (((bsdf.r_type() & (bsdf::type::diffuse | bsdf::type::glossy)) == 0)) {
						return reservoir;
					}
				}
				else {
					if (((bsdf.t_type() & (bsdf::type::diffuse | bsdf::type::glossy)) == 0)) {
						return reservoir;
					}
				}

				//���˕����̏����v�Z
				const direction lwo(-dir, y.ng(), y.ns());
				if (lwo.in_darkspot() || lwo.in_lower_hemisphere()) {
					return reservoir;
				}

				//�T���v���ɂ�����CW��ۊǂ���
				auto L = y.Le(lwo, lambda) * bsdf.f(dir) * dir.abs_cos() * lwo.abs_cos();
				float pdf = luminance(L);
				if (reservoir.get_M() != 0 && pdf != 0) {
					reservoir.set_W((reservoir.get_wsum() / reservoir.get_M()) / pdf);
				}

				if (scene.intersect(ray(isect.p(), dir, dist))) {
					reservoir.set_W(0);
				}

				return reservoir;
			}

			///////////////////////////////////////////////////////////////////////////////////////////////////

			////reservoir�̌���
			reservoir renderer::combine_reservoirs(const scene& scene, reservoir r[], int r_size, random_number_generator& rng, const wavelength& lambda, memory_allocator& allocator, int check, int biase_check) const {
				reservoir s;

				s.set_isect(r[0].get_isect());
				s.set_dir(r[0].get_dir());

				const auto isect = s.get_isect();
				//�q�b�g�_���Ȃ��ꍇ
				if (not(isect)) {
					return s;
				}

				//�q�b�g�_�����ߍގ������ʂ̏ꍇ
				if (isect.is_translucent() && isect.is_back_face()) {
					return s;
				}

				//��_�������̏ꍇ
				if (isect.is_emissive()) {
					return s;
				}

				//�o�˕����̏����擾
				const direction wo = s.get_dir();
				if (wo.in_darkspot()) {
					return s;
				}

				//BSDF���擾
				const auto bsdf = isect.bsdf(wo, lambda, allocator);
				if (bsdf.is_invalid()) {
					return s;
				}

				//r[0]�̃��U�[�o��s�Ɍ���
				float random = rng.generate_uniform_real(0.f);
				float Msum = 0;

				Msum += r[0].get_M();
				float pdf = 0;
				
				point_sample y = r[0].get_sample();
				vec4 tmp_ld = y.p() - isect.p();
				float dist2 = squared_norm(tmp_ld);
				float dist = sqrtf(dist2);
				auto dir = direction(tmp_ld / dist, isect.ng(), isect.ns());

				direction lwo(-dir, y.ng(), y.ns());

				spectrum L(lambda);
				if (!lwo.in_darkspot() && !lwo.in_lower_hemisphere()) {
					L = y.Le(lwo, lambda) * bsdf.f(dir) * dir.abs_cos() * lwo.abs_cos();
					pdf = luminance(L);
				}

				s.update(r[0].get_sample(), pdf * r[0].get_W() * r[0].get_M(), random);

				//�ߋ����U�[�o�̌����̏ꍇ
				if (check == 1) {
					//reservoir�ԂŃA�b�v�f�[�g���s��
					pdf = 0;
						
					y = r[1].get_sample();

					tmp_ld = y.p() - isect.p();
					dist2 = squared_norm(tmp_ld);
					dist = sqrtf(dist2);
					dir = direction(tmp_ld / dist, isect.ng(), isect.ns());

					spectrum L2(lambda);
					lwo = direction(-dir, y.ng(), y.ns());
					if (!lwo.in_darkspot() && !lwo.in_lower_hemisphere()) {
						L2 = y.Le(lwo, lambda) * bsdf.f(dir) * dir.abs_cos() * lwo.abs_cos();
						pdf = luminance(L2);
					}
					
					//�T���v�������ő�20�{�ŃN�����v����
					float clamp_size = 10;
					if (clamp_size > 20) {
						clamp_size = 20;
					}
					if (clamp_size < 1) {
						clamp_size = 1;
					}
					if (r[1].get_M() > (r[0].get_M() * clamp_size)) {
						r[1].set_M(r[0].get_M() * clamp_size);
					}
					Msum += r[1].get_M();
					s.update(r[1].get_sample(), pdf * r[1].get_W() * r[1].get_M(), random);
				}
				//�ߗ׃��U�[�o�̌����̏ꍇ
				else {
					//reservoir�ԂŃA�b�v�f�[�g���s��
					for (int i = 1; i < r_size; i++) {
						pdf = 0;
						
						y = r[i].get_sample();
						tmp_ld = y.p() - isect.p();
						dist2 = squared_norm(tmp_ld);
						dist = sqrtf(dist2);
						dir = direction(tmp_ld / dist, isect.ng(), isect.ns());

						spectrum L2(lambda);
						lwo = direction(-dir, y.ng(), y.ns());
						if (!lwo.in_darkspot() && !lwo.in_lower_hemisphere()) {
							L2 = y.Le(lwo, lambda) * bsdf.f(dir) * dir.abs_cos() * lwo.abs_cos();
							pdf = luminance(L2);
						}
						
						Msum += r[i].get_M();
						s.update(r[i].get_sample(), pdf * r[i].get_W() * r[i].get_M(), random);
					}
				}
				
				//M�̍��v�l���X�V
				s.set_M(Msum);

				y = s.get_sample();
				if (y.is_invalid()) {
					return s;
				}

				//���˕����̏����v�Z
				tmp_ld = y.p() - isect.p();
				dist2 = squared_norm(tmp_ld);
				dist = sqrtf(dist2);
				dir = direction(tmp_ld / dist, isect.ng(), isect.ns());

				if (dir.in_darkspot()) {
					return s;
				}
				else if (dir.in_upper_hemisphere()) {
					if (((bsdf.r_type() & (bsdf::type::diffuse | bsdf::type::glossy)) == 0)) {
						return s;
					}
				}
				else {
					if (((bsdf.t_type() & (bsdf::type::diffuse | bsdf::type::glossy)) == 0)) {
						return s;
					}
				}

				//���˕����̏����v�Z
				lwo = direction (-dir, y.ng(), y.ns());
				if (lwo.in_darkspot() || lwo.in_lower_hemisphere()) {
					return s;
				}

				//pdf�����߂�
				L = y.Le(lwo, lambda) * bsdf.f(dir) * dir.abs_cos() * lwo.abs_cos();
				pdf = luminance(L);

				//�o�C�A�X�̏ꍇ
				if (biase_check == 1) {
					//wsum, M, pdf���g��,W�����߂�
					if (s.get_M() != 0 && pdf != 0) {
						s.set_W((s.get_wsum() / s.get_M()) / pdf);
					}
				}
				//�A���o�C�A�X�̏ꍇ
				else {
					//�ereservoir�ɂ�����sample�ɂ��pdf��0���傫���Ȃ�Z�ɃT���v������������
					float Z = 0;
					
					for (int i = 0; i < r_size; i++) {
						const auto isect1 = r[i].get_isect();
						// �q�b�g�_���Ȃ��ꍇ
						if (not(isect1)) {
							continue;
						}

						//�q�b�g�_�����ߍގ������ʂ̏ꍇ
						if (isect1.is_translucent() && isect1.is_back_face()) {
							continue;
						}

						//��_�������̏ꍇ
						if (isect1.is_emissive()) {
							continue;
						}

						//���̏o�˕����̏����v�Z
						const auto dir1 = r[i].get_dir();
						if (dir1.in_darkspot()) {
							continue;
						}

						//���̈ʒu�ł̏o�˕����̏����v�Z
						const direction wo1(dir1, isect1.ng(), isect1.ns());
						if (wo1.in_darkspot()) {
							continue;
						}

						//BSDF���擾
						const auto bsdf1 = isect1.bsdf(wo1, lambda, allocator);
						if (bsdf1.is_invalid()) {
							continue;
						}

						//���˕����̏����v�Z
						vec4 tmp_ld1 = y.p() - isect1.p();
						float dist3 = squared_norm(tmp_ld1);
						float dist1 = sqrtf(dist3);
						auto dir2 = direction(tmp_ld1 / dist1, isect1.ng(), isect1.ns());
						if (dir2.in_darkspot()) {
							continue;
						}
						else if (dir2.in_upper_hemisphere()) {
							if (((bsdf1.r_type() & (bsdf::type::diffuse | bsdf::type::glossy)) == 0)) {
								continue;
							}
						}
						else {
							if (((bsdf1.t_type() & (bsdf::type::diffuse | bsdf::type::glossy)) == 0)) {
								continue;
							}
						}

						//���˕����̏����v�Z
						auto lwo1 = direction(-dir2, y.ng(), y.ns());
						if (lwo1.in_darkspot() || lwo1.in_lower_hemisphere()) {
							continue;
						}

						auto L2 = y.Le(lwo1, lambda) * bsdf1.f(dir2) * dir2.abs_cos() * lwo1.abs_cos();
						float pdf2 = luminance(L2);
						if (pdf2 > 0) {
							Z += r[i].get_M();
						}
					}

					float m = 0;
					if (Z != 0) {
						m = 1.f / Z;
					}

					//wsum, m, pdf���g��,W�����߂�
					if (pdf != 0) {
						s.set_W((s.get_wsum() * m) / pdf);
					}
				}
				
				
				return s;
			}

			///////////////////////////////////////////////////////////////////////////////////////////////////
			////�s�N�Z�����Ƃ�reservoir�����߂�

			reservoir renderer::RIS_pixel(const scene& scene, const wavelength& lambda, ray r, random_number_generator& rng, memory_allocator& allocator) const
			{
				reservoir reservoir;
				const auto isect = scene.intersection(r);

				reservoir.set_dir(-r.d());

				reservoir.set_isect(isect);

				//�q�b�g�_������ꍇ
				if (isect) {
					//�q�b�g�_�����ߍގ������ʂ̏ꍇ
					if (isect.is_translucent() && isect.is_back_face()) {
						return reservoir;
					}

					//�o�˕����̏����v�Z
					const direction wo(-r.d(), isect.ng(), isect.ns());
					if (wo.in_darkspot()) {
						return reservoir;
					}

					//��_�������̏ꍇ
					if (isect.is_emissive()) {
						return reservoir;
					}

					//BSDF���擾
					const auto bsdf = isect.bsdf(wo, lambda, allocator);
					if (bsdf.is_invalid()) {
						return reservoir;
					}
					
					//RIS�v�Z
					if (scene.area_lights().Pv() > 0) {
						reservoir = WRS_RIS(scene, isect, bsdf, lambda, scene.area_lights(), rng);
						
						reservoir.set_dir(wo);

						reservoir.set_isect(isect);
					}
				}

				return reservoir;
			}

			///////////////////////////////////////////////////////////////////////////////////////////////////

			////�s�N�Z�����Ƃ̋P�x�����߂�
			inline spectrum renderer::radiance_pixel(reservoir& reservoir, const scene& scene, const wavelength& lambda, memory_allocator& allocator) const
			{
				spectrum L(lambda);
				
				const auto isect = reservoir.get_isect();
				//�q�b�g�_���Ȃ��ꍇ
				if (not(isect)) {
					if (scene.environment().Pv() > 0) {
						L += scene.environment().L(reservoir.get_dir(), lambda);
					}
				}
				else {
					//�q�b�g�_�����ߍގ������ʂ̏ꍇ
					if (isect.is_translucent() && isect.is_back_face()) {
						return L;
					}

					//�o�˕����̏����擾����
					const direction wo = reservoir.get_dir();
					if (wo.in_darkspot()) {
						return L;
					}

					//��_�������̏ꍇ
					if (isect.is_emissive()) {
						if (isect.is_front_face()) {
							L += spectrum(1, lambda) * isect.Le(wo, lambda);
						}
						return L;
					}

					//BSDF���擾����
					const auto bsdf = isect.bsdf(wo, lambda, allocator);
					if (bsdf.is_invalid()) {
						return L;
					}

					//�T���v�����擾����
					const auto lsample = reservoir.get_sample();
					if (lsample.is_invalid()) {
						return L;
					}

					//���˕����̏����v�Z
					const vec4 tmp_ld = lsample.p() - isect.p();
					const float dist2 = squared_norm(tmp_ld);
					const float dist = sqrtf(dist2);
					const auto wi = direction(tmp_ld / dist, isect.ng(), isect.ns());
					if (wi.in_darkspot()) {
						return L;
					}
					else if (wi.in_upper_hemisphere()) {
						if ((bsdf.r_type() & (bsdf::type::diffuse | bsdf::type::glossy)) == 0) { return L; }
					}
					else {
						if ((bsdf.t_type() & (bsdf::type::diffuse | bsdf::type::glossy)) == 0) { return L; }
					}
					
					//���˕����̏����v�Z
					const direction lwo(-wi, lsample.ng(), lsample.ns());
					if (lwo.in_darkspot() || lwo.in_lower_hemisphere()) {
						return L;
					}

					//�P�x�v�Z
					if (not(scene.intersect(ray(isect.p(), wi, dist)))) {
						L += (lsample.Le(lwo, lambda) * bsdf.f(wi) * wi.abs_cos() * lwo.abs_cos() / dist2) * reservoir.get_W();
					}

				}
				return L;
			}

			///////////////////////////////////////////////////////////////////////////////////////////////////

		} //namespace ReSTIR_light

		///////////////////////////////////////////////////////////////////////////////////////////////////

	} //namespace rndr

	///////////////////////////////////////////////////////////////////////////////////////////////////

} //namespace ns

///////////////////////////////////////////////////////////////////////////////////////////////////
