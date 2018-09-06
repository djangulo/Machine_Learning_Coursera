function [bestC, bestSigma] = findBestParams()

load('ex6data3.mat');

% Try different SVM Parametrs here


ranges = [0.01 0.03 0.1 0.3 1.0 3.0 10 30];

bestC = 0;
bestSigma = 0;
bestError = 1;

for cc = ranges
    for ssigma = ranges
        model = svmTrain(X, y, cc, @(x1, x2) gaussianKernel(x1, x2, ssigma));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        fprintf(['For C = %f, sigma = %f, error is %f\n\n'], cc, ssigma,err);

        if err < bestError
            bestError = err;
            bestC = cc;
            bestSigma = ssigma;
            fprintf(['\nFound lowest yet! error: %f, sigma: %f, C: %f'], bestError, bestC, bestSigma);
        end

    end
end


end