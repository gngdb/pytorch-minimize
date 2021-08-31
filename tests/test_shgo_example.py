from scipy.optimize import rosen, shgo, brute

if __name__ == '__main__':
    bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
    
    minimizer_args = dict(method='SLSQP', options={'disp':True, 'maxiter':10000})
    shgo_kwargs = dict(bounds=bounds, options={'disp':True})

    result = shgo(rosen, minimizer_kwargs=minimizer_args, **shgo_kwargs)
    ranges=[slice(a,b,0.25) for a,b in bounds]
    brute_kwargs = {'ranges': ranges}
    #result = brute(rosen, **brute_kwargs)

    # print(result)
    print(result.x, result.fun)
