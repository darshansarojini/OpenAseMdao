from casadi import *
import math

class Integrator:
    def __init__(self, tolerance, model, bdf_order, inputs, outputs):
        if model.options['num_timesteps'] == 1:
            k_lim = 500  # Limits to make sure the system does not get stuck in an infinite convergence loop
            k_start = 0
            newton_iteration_counter = 0
            newton_iteration_convergence_flag = True

            outputs['x'] = model.numeric_storage['x0']

            while newton_iteration_convergence_flag:
                k_start = k_start + 1
                if k_start > k_lim:
                    print('Newton iteration limit reached')
                    break
                Fty = model.symbolic_functions['Residuals'](outputs['x'], np.zeros((model.numeric_storage['x0'].shape[0], 1)), inputs['Xac'], inputs['forces_dist'],
                                                            inputs['moments_dist'], inputs['forces_conc'], inputs['moments_conc'], inputs['cs'])
                Jac = model.symbolic_functions['Jacobian'](outputs['x'], np.zeros((model.numeric_storage['x0'].shape[0], 1)), inputs['Xac'], inputs['forces_dist'],
                                                           inputs['moments_dist'], inputs['forces_conc'], inputs['moments_conc'], inputs['cs'])

                dRdX = Jac[0:model.symbolic_expressions['Residual'].shape[0], 0:model.symbolic_expressions['Residual'].shape[0]]

                delta_x_vec = np.linalg.solve(dRdX, -Fty)

                newton_iteration_convergence_flag = False

                norm_res = np.linalg.norm(Fty)

                if norm_res > tolerance:  # The process is not yet done
                    newton_iteration_convergence_flag = True
                if newton_iteration_convergence_flag == 0:
                    break  # we are done
                else:
                    # new guess
                    outputs['x'] = outputs['x'] + delta_x_vec
                newton_iteration_counter = newton_iteration_counter + 1
        else:
            outputs['x'][:, 0] = model.numeric_storage['x0']
            model.numeric_storage['xDot'][:, 0] = np.zeros(model.numeric_storage['x0'].shape[0])
            # Run External Loop:
            for iteration in range(1, model.numeric_storage['max_step']):
                # print('==================================================================')
                # print(iteration)
                model.numeric_storage['current_step'] = iteration
                model.numeric_storage['time'][iteration] = model.numeric_storage['time'][iteration - 1] + \
                                                           model.options['time_step']
                model.numeric_storage['x_minus_1'] = outputs['x'][:, iteration - 1]
                model.numeric_storage['x_local'] = outputs['x'][:, iteration - 1]
                if iteration >= 2:
                    model.numeric_storage['x_minus_2'] = outputs['x'][:, iteration - 2]

                # Now figuring out constants for time marching scheme:
                if bdf_order == 1:
                    model.numeric_storage['k0'] = 1 / model.options['time_step']
                    model.numeric_storage['k1'] = -model.numeric_storage['k0']
                    model.numeric_storage['xDot_local'] = model.numeric_storage['k0'] * model.numeric_storage['x_local'] + \
                                                               model.numeric_storage[
                                                                   'k1'] * model.numeric_storage['x_minus_1']
                if bdf_order == 2:
                    if iteration >= 2:
                        model.numeric_storage['k0'] = (3 / 2) * (1 / model.options['time_step'])
                        model.numeric_storage['k1'] = -2 * (1 / model.options['time_step'])
                        model.numeric_storage['k2'] = (1 / 2) * (1 / model.options['time_step'])
                        model.numeric_storage['xDot_local'] = model.numeric_storage['k0'] * model.numeric_storage['x_local'] + \
                                                              model.numeric_storage[
                                                                       'k1'] * model.numeric_storage['x_minus_1'] + \
                                                                          model.numeric_storage['k2'] * \
                                                                          model.numeric_storage[
                                                                       'x_minus_2']
                    else:
                        model.numeric_storage['k0'] = 1 / model.options['time_step']
                        model.numeric_storage['k1'] = -model.numeric_storage['k0']
                        model.numeric_storage['xDot_local'] = model.numeric_storage['k0'] * model.numeric_storage['x_local'] + \
                                                              model.numeric_storage['k1'] * \
                                                              model.numeric_storage['x_minus_1']

                k_lim = 500  # Limits to make sure the system does not get stuck in an infinite convergence loop
                k_start = 0
                model.update_time_varying_quantities(model.numeric_storage['x_local'], model.numeric_storage['xDot_local'], iteration)
                newton_iteration_counter = 0
                newton_iteration_convergence_flag = True
                while newton_iteration_convergence_flag:
                    k_start = k_start + 1
                    if k_start > k_lim:
                        print('Newton iteration limit reached')
                        break
                    Fty = model.symbolic_functions['Residuals'](model.numeric_storage['x_local'], model.numeric_storage['xDot_local'],
                                                                model.numeric_storage['Xac'], model.numeric_storage['forces_dist'],
                                                                model.numeric_storage['moments_dist'], model.numeric_storage['forces_conc'],
                                                                model.numeric_storage['moments_conc'], inputs['cs'])

                    Jac = model.symbolic_functions['Jacobian'](model.numeric_storage['x_local'], model.numeric_storage['xDot_local'],
                                                               model.numeric_storage['Xac'],
                                                               model.numeric_storage['forces_dist'],
                                                               model.numeric_storage['moments_dist'], model.numeric_storage['forces_conc'],
                                                               model.numeric_storage['moments_conc'], inputs['cs'])

                    dRdX = Jac[0:model.symbolic_expressions['Residual'].shape[0],
                           0:model.symbolic_expressions['Residual'].shape[0]]

                    dRdXdot = Jac[0:model.symbolic_expressions['Residual'].shape[0],
                              model.symbolic_expressions['Residual'].shape[0]:2 *
                                                                              model.symbolic_expressions['Residual'].shape[0]]

                    delta_x_vec = np.linalg.solve((dRdX + model.numeric_storage['k0'] * dRdXdot), -Fty)

                    newton_iteration_convergence_flag = False

                    norm_res = np.linalg.norm(Fty)

                    if norm_res > tolerance:  # The process is not yet done
                        newton_iteration_convergence_flag = True
                    if newton_iteration_convergence_flag == 0:
                        break  # we are done
                    else:
                        model.numeric_storage['x_local'] = model.numeric_storage['x_local'] + np.squeeze(delta_x_vec)
                        if bdf_order == 1:
                            model.numeric_storage['xDot_local'] = model.numeric_storage['k0'] * model.numeric_storage['x_local'] + \
                                                                  model.numeric_storage[
                                                                     'k1'] * model.numeric_storage['x_minus_1']
                        if bdf_order == 2:
                            if model.numeric_storage['current_step'] >= 2:
                                model.numeric_storage['xDot_local'] = model.numeric_storage['k0'] * model.numeric_storage['x_local'] + \
                                                                      model.numeric_storage['k1'] * model.numeric_storage[
                                                                         'x_minus_1'] + model.numeric_storage['k2'] * \
                                                                      model.numeric_storage['x_minus_2']
                            else:
                                model.numeric_storage['xDot_local'] = model.numeric_storage['k0'] * model.numeric_storage['x_local'] + \
                                                                     model.numeric_storage['k1'] * model.numeric_storage[
                                                                         'x_minus_1']
                    newton_iteration_counter = newton_iteration_counter + 1

                # Record dynamics
                outputs['x'][:, iteration] = model.numeric_storage['x_local']
                model.numeric_storage['xDot'][:, iteration] = model.numeric_storage['xDot_local']