#define FLOAT_TYPE float
// Rho Aias!!!
	
int correlate2d(
		FLOAT_TYPE* z,
		int batchs,
		int z_h,
		int z_w,
		int i_c,
		int h_step,
		int w_step,
		FLOAT_TYPE* f,
		int f_h,
		int f_w,
		int o_c,
		FLOAT_TYPE* o,
		int o_h,
		int o_w
	) {
	/*
	enumarate dims
	b,oh,ow,ih,iw,ic,oc,
	*/
	int o_b_size = o_h*o_w*o_c;
	int o_oh_size = o_w*o_c;
	int f_fh_size = f_w*i_c*o_c;
	int f_fw_size = i_c*o_c;
	int z_b_size = z_h*z_w*i_c;
	int z_oh_size = h_step*z_w*i_c;
	int z_ow_size = w_step*i_c;
	int z_fh_size = z_w*i_c;

	//std::cout << "Receive" << std::endl;
	FLOAT_TYPE* o_b = o;
	FLOAT_TYPE* z_b = z;
	for (int b = 0; b < batchs; ++b) {
		FLOAT_TYPE* o_oh = o_b;
		FLOAT_TYPE* z_oh = z_b;
		for (int oh = 0; oh < o_h; ++oh) {

			FLOAT_TYPE* o_ow = o_oh;
			FLOAT_TYPE* z_ow = z_oh;
			for (int ow = 0; ow < o_w; ++ow) {

				FLOAT_TYPE* f_fh = f;
				FLOAT_TYPE* z_fh = z_ow;
				for (int fh = 0; fh < f_h; ++fh) {
					FLOAT_TYPE* f_fw = f_fh;
					FLOAT_TYPE* z_fw = z_fh;
					for (int fw = 0; fw < f_w; ++fw) {

						FLOAT_TYPE* f_ic = f_fw;
						for (int ic = 0; ic < i_c; ++ic) {
							for (int oc = 0; oc < o_c; ++oc) {
								// o[b][oh][ow][oc] += z[b][ih][iw][ic] * f[fh][fw][ic][oc]
								o_ow[oc] += z_fw[ic] * f_ic[oc];
								// std::cout << (o_ow - o) + oc << '=' << (z_fw - z) + ic << '+' << (f_ic - f) + oc << std::endl;
							}
							f_ic += o_c;
						}

						f_fw += f_fw_size;
						z_fw += i_c;
					}

					f_fh += f_fh_size;
					z_fh += z_fh_size;
				}

				o_ow += o_c;
				z_ow += z_ow_size;
			}
			o_oh += o_oh_size;
			z_oh += z_oh_size;
		}
		o_b += o_b_size;
		z_b += z_b_size;
	}
	//std::cout << "Receive" << std::endl;
	return 0;
}

int conv2d_filter_gradient(
		FLOAT_TYPE* z,
		int batchs,
		int z_h,
		int z_w,
		int i_c,
		FLOAT_TYPE* f,
		int f_h,
		int f_w,
		int o_c,
		FLOAT_TYPE* o,
		int o_h,
		int o_w
	) {
	/*
		enumarate dims
		b, o_h, o_w, f_h, f_w, i_c, o_c
	*/
	int o_oh_size = o_w*i_c*o_c;
	int o_ow_size = i_c*o_c;
	int z_b_size = z_h*z_w*i_c;
	int z_oh_size = z_w*i_c;
	int z_fh_size = z_w*i_c;
	int f_b_size = f_h*f_w*o_c;
	int f_fh_size = f_w*o_c;

	FLOAT_TYPE* z_b = z;
	FLOAT_TYPE* f_b = f;
	for (int b = 0; b < batchs; ++b) {
		FLOAT_TYPE* o_oh = o;
		FLOAT_TYPE* z_oh = z_b;
		for (int oh = 0; oh < o_h; ++oh) {
			FLOAT_TYPE* o_ow = o_oh;
			FLOAT_TYPE* z_ow = z_oh;
			for (int ow = 0; ow < o_w; ++ow) {
				FLOAT_TYPE* z_fh = z_ow;
				FLOAT_TYPE* f_fh = f_b;
				for (int fh = 0; fh < f_h; ++fh) {
					FLOAT_TYPE* z_fw = z_fh;
					FLOAT_TYPE* f_fw = f_fh;
					for (int fw = 0; fw < f_w; ++fw) {
						FLOAT_TYPE* o_ic = o_ow;
						for (int ic = 0; ic < i_c; ++ic) {
							for (int oc = 0; oc < o_c; ++oc) {
								// o[oh][ow][ic][oc] += z[b][oh+fh][ow+fw][ic] * f[b][fh][fw][oc]
								o_ic[oc] += z_fw[ic] * f_fw[oc];
								// std::cout << (o_ic - o) + oc << '=' << (z_fw - z) + ic << '+' << (f_fw - f) + oc << std::endl;
							}
							o_ic += o_c;
						}
						z_fw += i_c;
						f_fw += o_c;
					}	
					z_fh += z_fh_size;
					f_fh += f_fh_size;
				}
				o_ow += o_ow_size;
				z_ow += i_c;
			}
			o_oh += o_oh_size;
			z_oh += z_oh_size;
		}
		z_b += z_b_size;
		f_b += f_b_size;
	}
	return 0;
}

int max_pool_gradient(
		FLOAT_TYPE* g,
		int batchs,
		int g_h,
		int g_w,
		int i_c,
		FLOAT_TYPE* o,
		int h_step,
		int w_step,
		FLOAT_TYPE* z,
		int z_h,
		int z_w
	) {
	int g_b_size = g_h*g_w*i_c;
	int g_gh_size = g_w*i_c;
	int z_b_size = z_h*z_w*i_c;
	int z_gh_size = h_step*z_w*i_c;
	int z_gw_size = w_step*i_c;
	int z_h_size = z_w*i_c;

	for (int b = 0; b < batchs; ++b) {
		FLOAT_TYPE* g_b = g + b*g_b_size;
		FLOAT_TYPE* z_b = z + b*z_b_size;

		for (int gh = 0; gh < g_h; ++gh) {
			FLOAT_TYPE* g_gh = g_b + gh*g_gh_size;
			FLOAT_TYPE* z_gh = z_b + gh*z_gh_size;

			for (int gw = 0; gw < g_w; ++gw) {
				FLOAT_TYPE* g_gw = g_gh + gw * i_c;
				FLOAT_TYPE* z_gw = z_gh + gw * z_gw_size;

				for (int ic = 0; ic < i_c; ++ic) {
					FLOAT_TYPE* z_ic = z_gw + ic;
					FLOAT_TYPE* max_loc = z_ic;

					for (int h = 0; h < h_step; ++h) {
						FLOAT_TYPE* z_h = z_ic + h*z_h_size;

						for (int w = 0; w < w_step; ++w) {
							FLOAT_TYPE* z_w = z_h + w*i_c;

							if ((*z_w) > (*max_loc)) {
								max_loc = z_w;
							}
						}
					}
					o[max_loc - z] += g_gw[ic];
				}
			}
		}
	}
	return 0;
}
