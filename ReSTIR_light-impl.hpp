
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
			//前方宣言
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

			//コンストラクタ
			inline renderer::renderer(const scene&, const camera&, const size_t spp, const size_t nt)
			{
				m_spp = spp;
				m_nt = nt;
			}

			///////////////////////////////////////////////////////////////////////////////////////////////////

			//レンダリング
			inline imagef renderer::render(const scene& scene, const camera& camera) const
			{
				//スクリーンの初期化
				const int w = camera.res_x();
				const int h = camera.res_y();
				imagef screen(w, h, 3);
				int biase_check = 1;

				//biase結合
				if (biase_check == 1) {
					//過去フレームのreservoirを保存する変数
					static reservoir reservoir_p[512][512];

					//risを行うreservoirを宣言
					reservoir** reservoir_ris = new reservoir * [w];

					//temporal reuseを行うreservoirを宣言
					reservoir** reservoir_t = new reservoir * [w];

					reservoir** reservoir_t2 = new reservoir * [w];

					for (int i = 0; i < w; i++) {
						reservoir_ris[i] = new reservoir[h];

						reservoir_t[i] = new reservoir[h];

						reservoir_t2[i] = new reservoir[h];
					}

					//ピクセルごとのreservoirを求める
					in_parallel(std::make_tuple(w, h), [&, this](const std::tuple<int, int>& idx)
						{
							const int x = std::get<0>(idx);
							const int y = std::get<1>(idx);

							thread_local pseudo_rng rng(
								std::random_device{}()
							);
							thread_local memory_allocator allocator;

							//RIS計算
							reservoir_ris[x][y] = RIS_pixel(scene, wavelength(rng), camera.sample(x, y, rng), rng, allocator);

							//過去フレームのreservoirと結合
							reservoir r[2];
							r[0] = reservoir_ris[x][y];
							r[1] = reservoir_p[x][y];
							
							reservoir_t[x][y] = combine_reservoirs(scene, r, 2, rng, wavelength(rng), allocator, 1, biase_check);
						}, m_nt);

					//spatial reuseでreservoirの結合
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
								//近隣リザーバの収集
								reservoir r[6];
								int r_size = sizeof(r) / sizeof(reservoir);
								r[0] = reservoir_t[x][y];

								if (r[0].get_isect()) {
									const auto ng1 = r[0].get_isect().ng();
									int count = 1;

									//法線ベクトル同士の角度が一定以下の場合のみ，結合するリザーバとする
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
								//２回目の近隣リザーバの収集
								reservoir r[6];
								int r_size = sizeof(r) / sizeof(reservoir);
								r[0] = reservoir_t2[x][y];

								if (r[0].get_isect()) {
									const auto ng1 = r[0].get_isect().ng();
									int count = 1;

									//法線ベクトル同士の角度が一定以下の場合のみ，結合するリザーバとする
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

								//画像の輝度を求める
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

								//重みの割合に応じて，輝度の加重平均をとる
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
						//risのreservoirのメモリを解放
						delete[] reservoir_ris[i];

						//temporal reservoirのメモリを解放
						delete[] reservoir_t[i];

						delete[] reservoir_t2[i];
					}

					delete[] reservoir_ris;

					delete[] reservoir_t;

					delete[] reservoir_t2;

				}
				//unbiaseの結合
				else {
					//過去フレームのreservoirを保存する変数
					static reservoir reservoir_p[512][512];

					//risを行うreservoirを宣言
					reservoir** reservoir_ris = new reservoir * [w];

					//temporal reuseを行うreservoirを宣言
					reservoir** reservoir_t = new reservoir * [w];

					for (int i = 0; i < w; i++) {
						reservoir_ris[i] = new reservoir[h];

						reservoir_t[i] = new reservoir[h];
					}

					//ピクセルごとのreservoirを求める
					in_parallel(std::make_tuple(w, h), [&, this](const std::tuple<int, int>& idx)
						{
							const int x = std::get<0>(idx);
							const int y = std::get<1>(idx);

							thread_local pseudo_rng rng(
								std::random_device{}()
							);
							thread_local memory_allocator allocator;

							//RIS計算
							reservoir_ris[x][y] = RIS_pixel(scene, wavelength(rng), camera.sample(x, y, rng), rng, allocator);

							//過去フレームのreservoirと結合
							reservoir r[2];
							r[0] = reservoir_ris[x][y];
							r[1] = reservoir_p[x][y];

							reservoir_t[x][y] = combine_reservoirs(scene, r, 2, rng, wavelength(rng), allocator, 1, 1);

						}, m_nt);

					//spatial reuseでreservoirの結合
					in_parallel(std::make_tuple(w, h), [&, this](const std::tuple<int, int>& idx)
						{
							const int x = std::get<0>(idx);
							const int y = std::get<1>(idx);

							thread_local pseudo_rng rng(
								std::random_device{}()
							);
							thread_local memory_allocator allocator;

							//近隣リザーバの収集
							reservoir r[4];
							int r_size = sizeof(r) / sizeof(reservoir);
							r[0] = reservoir_t[x][y];

							if (r[0].get_isect()) {
								const auto ng1 = r[0].get_isect().ng();
								int count = 1;

								//法線ベクトル同士の角度が一定以下の場合のみ，結合するリザーバとする
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

							//重みの割合に応じて，輝度の加重平均をとる
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
						//risのreservoirのメモリを解放
						delete[] reservoir_ris[i];

						//temporal reservoirのメモリを解放
						delete[] reservoir_t[i];
					}

					delete[] reservoir_ris;

					delete[] reservoir_t;
				}

				return screen;
			}

			///////////////////////////////////////////////////////////////////////////////////////////////////

			//WRSを用いたRIS計算
			reservoir renderer::WRS_RIS(const scene& scene, const intersection& isect, const bsdf& bsdf, const wavelength& lambda, area_lights const& area_lights, random_number_generator& rng) const {
				float random = rng.generate_uniform_real(0.f);
				reservoir reservoir;
				
				//光源のサンプルからpdfを求めて，reservoirの更新を行う
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

				//サンプルが無効な値ならWを0にする
				const auto y = reservoir.get_sample();
				if (y.is_invalid()) {
					return reservoir;
				}

				//入射方向の情報を計算
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

				//放射方向の情報を計算
				const direction lwo(-dir, y.ng(), y.ns());
				if (lwo.in_darkspot() || lwo.in_lower_hemisphere()) {
					return reservoir;
				}

				//サンプルにおける，Wを保管する
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

			////reservoirの結合
			reservoir renderer::combine_reservoirs(const scene& scene, reservoir r[], int r_size, random_number_generator& rng, const wavelength& lambda, memory_allocator& allocator, int check, int biase_check) const {
				reservoir s;

				s.set_isect(r[0].get_isect());
				s.set_dir(r[0].get_dir());

				const auto isect = s.get_isect();
				//ヒット点がない場合
				if (not(isect)) {
					return s;
				}

				//ヒット点が透過材質かつ裏面の場合
				if (isect.is_translucent() && isect.is_back_face()) {
					return s;
				}

				//交点が光源の場合
				if (isect.is_emissive()) {
					return s;
				}

				//出射方向の情報を取得
				const direction wo = s.get_dir();
				if (wo.in_darkspot()) {
					return s;
				}

				//BSDFを取得
				const auto bsdf = isect.bsdf(wo, lambda, allocator);
				if (bsdf.is_invalid()) {
					return s;
				}

				//r[0]のリザーバをsに結合
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

				//過去リザーバの結合の場合
				if (check == 1) {
					//reservoir間でアップデートを行う
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
					
					//サンプル数を最大20倍でクランプする
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
				//近隣リザーバの結合の場合
				else {
					//reservoir間でアップデートを行う
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
				
				//Mの合計値を更新
				s.set_M(Msum);

				y = s.get_sample();
				if (y.is_invalid()) {
					return s;
				}

				//入射方向の情報を計算
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

				//放射方向の情報を計算
				lwo = direction (-dir, y.ng(), y.ns());
				if (lwo.in_darkspot() || lwo.in_lower_hemisphere()) {
					return s;
				}

				//pdfを求める
				L = y.Le(lwo, lambda) * bsdf.f(dir) * dir.abs_cos() * lwo.abs_cos();
				pdf = luminance(L);

				//バイアスの場合
				if (biase_check == 1) {
					//wsum, M, pdfを使い,Wを求める
					if (s.get_M() != 0 && pdf != 0) {
						s.set_W((s.get_wsum() / s.get_M()) / pdf);
					}
				}
				//アンバイアスの場合
				else {
					//各reservoirにおけるsampleによるpdfが0より大きいならZにサンプル数を加える
					float Z = 0;
					
					for (int i = 0; i < r_size; i++) {
						const auto isect1 = r[i].get_isect();
						// ヒット点がない場合
						if (not(isect1)) {
							continue;
						}

						//ヒット点が透過材質かつ裏面の場合
						if (isect1.is_translucent() && isect1.is_back_face()) {
							continue;
						}

						//交点が光源の場合
						if (isect1.is_emissive()) {
							continue;
						}

						//元の出射方向の情報を計算
						const auto dir1 = r[i].get_dir();
						if (dir1.in_darkspot()) {
							continue;
						}

						//この位置での出射方向の情報を計算
						const direction wo1(dir1, isect1.ng(), isect1.ns());
						if (wo1.in_darkspot()) {
							continue;
						}

						//BSDFを取得
						const auto bsdf1 = isect1.bsdf(wo1, lambda, allocator);
						if (bsdf1.is_invalid()) {
							continue;
						}

						//入射方向の情報を計算
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

						//放射方向の情報を計算
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

					//wsum, m, pdfを使い,Wを求める
					if (pdf != 0) {
						s.set_W((s.get_wsum() * m) / pdf);
					}
				}
				
				
				return s;
			}

			///////////////////////////////////////////////////////////////////////////////////////////////////
			////ピクセルごとのreservoirを求める

			reservoir renderer::RIS_pixel(const scene& scene, const wavelength& lambda, ray r, random_number_generator& rng, memory_allocator& allocator) const
			{
				reservoir reservoir;
				const auto isect = scene.intersection(r);

				reservoir.set_dir(-r.d());

				reservoir.set_isect(isect);

				//ヒット点がある場合
				if (isect) {
					//ヒット点が透過材質かつ裏面の場合
					if (isect.is_translucent() && isect.is_back_face()) {
						return reservoir;
					}

					//出射方向の情報を計算
					const direction wo(-r.d(), isect.ng(), isect.ns());
					if (wo.in_darkspot()) {
						return reservoir;
					}

					//交点が光源の場合
					if (isect.is_emissive()) {
						return reservoir;
					}

					//BSDFを取得
					const auto bsdf = isect.bsdf(wo, lambda, allocator);
					if (bsdf.is_invalid()) {
						return reservoir;
					}
					
					//RIS計算
					if (scene.area_lights().Pv() > 0) {
						reservoir = WRS_RIS(scene, isect, bsdf, lambda, scene.area_lights(), rng);
						
						reservoir.set_dir(wo);

						reservoir.set_isect(isect);
					}
				}

				return reservoir;
			}

			///////////////////////////////////////////////////////////////////////////////////////////////////

			////ピクセルごとの輝度を求める
			inline spectrum renderer::radiance_pixel(reservoir& reservoir, const scene& scene, const wavelength& lambda, memory_allocator& allocator) const
			{
				spectrum L(lambda);
				
				const auto isect = reservoir.get_isect();
				//ヒット点がない場合
				if (not(isect)) {
					if (scene.environment().Pv() > 0) {
						L += scene.environment().L(reservoir.get_dir(), lambda);
					}
				}
				else {
					//ヒット点が透過材質かつ裏面の場合
					if (isect.is_translucent() && isect.is_back_face()) {
						return L;
					}

					//出射方向の情報を取得する
					const direction wo = reservoir.get_dir();
					if (wo.in_darkspot()) {
						return L;
					}

					//交点が光源の場合
					if (isect.is_emissive()) {
						if (isect.is_front_face()) {
							L += spectrum(1, lambda) * isect.Le(wo, lambda);
						}
						return L;
					}

					//BSDFを取得する
					const auto bsdf = isect.bsdf(wo, lambda, allocator);
					if (bsdf.is_invalid()) {
						return L;
					}

					//サンプルを取得する
					const auto lsample = reservoir.get_sample();
					if (lsample.is_invalid()) {
						return L;
					}

					//入射方向の情報を計算
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
					
					//放射方向の情報を計算
					const direction lwo(-wi, lsample.ng(), lsample.ns());
					if (lwo.in_darkspot() || lwo.in_lower_hemisphere()) {
						return L;
					}

					//輝度計算
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
