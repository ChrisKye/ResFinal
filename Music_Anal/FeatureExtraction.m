myDir = '../Data/EDIT_Music'; %gets directory
myFiles = dir(fullfile(myDir,'*.wav'));
df2 = struct;

for k = 1:40                                   %% test using 1:1
    baseName = strcat('../Data/EDIT_Music/', myFiles(k).name);

    %%Dynamics, 1
    rms = mirrms(baseName);
    rms = get(rms, 'Data');
    df(k).rms = cell2mat(rms{1});
    'completed Dynamics'                        %% rms energy

    %%Rhythm, 6
    temp = mirfluctuation(baseName,'Summary');

    dat = get(temp, 'Data');
    dat = dat{1};
    df(k).fluctuationPeak = max(dat{1});        %%fluctuation max

    centroid = mircentroid(temp,'MaxEntropy',0);
    centroid = get(centroid, 'Data');
    df(k).fluctuationCentroid = cell2mat(centroid{1}{1});     %%fluctuation centroid

    tempo = mirtempo(baseName);
    tempo = get(tempo, 'Data');
    tempo = tempo{1};
    df(k).tempo = cell2mat(tempo{1});           %%tempo

    pul = mirpulseclarity(baseName);
    pul = get(pul, 'Data');
    pul = pul{1};
    df(k).pulseClarity = cell2mat(pul);         %%pulse clarity

    attackTime = mirattacktime(baseName);
    attackTime = get(attackTime, 'Data');
    df(k).meanAttackTime = mean(attackTime{1}{1}); %%mean attack time

    attackSlope = mirattackslope(baseName);
    attackSlope = get(attackSlope, 'Data');
    df(k).meanAttackSlope = mean(attackSlope{1}{1}); %%mean attack slope

    'completed Rhythm'

    %%Timbre, 20 (7 + mfcc)
    zerox = mirzerocross(baseName);
    zerox = get(zerox, 'Data');
    zerox = zerox{1}
    df(k).zeroCross = zerox{1};                 %% zero crossing rate

    spec = mircentroid(baseName);
    spec = get(spec, 'Data');
    spec = spec{1};
    df(k).spectralCentroid = spec{1}{1};           %%spectral centroid

    spr = mirspread(baseName);
    spr = get(spr, 'Data');
    spr = spr{1};
    df(k).spectralSpread = spr{1}{1};              %%spectral spread

    skew = mirskewness(baseName);
    skew = get(skew, 'Data');
    skew = skew{1}
    df(k).spectralSkewness = skew{1}{1};            %% spectral skewness

    kurt = mirkurtosis(baseName);
    kurt = get(kurt, 'Data');
    kurt = kurt{1}
    df(k).spectralKurtosis = kurt{1}{1};           %% spectral kurtosis

    flat = mirflatness(baseName);
    flat = get(flat, 'Data');
    flat = flat{1};
    df(k).spectralFlatness = flat{1};           %% spectral flatness

    ent = mirentropy(baseName);
    ent = get(ent, 'Data');
    ent = ent{1};
    df(k).spectralEntropy = ent{1};                %% spectral entropy
    'completed Timbre'

    mfcc = mirmfcc(baseName);
    mfcc = get(mfcc, 'Data');
    mfcc = mfcc{1};
    df(k).mfcc = mfcc{1};                       %% mfcc 13 coefficients

    %%Harmony, 4
    x = mirhcdf(baseName);
    dat = get(x, 'Data');
    dat = dat{1};
    df(k).harmonicChange = mean(dat{1});

    [key,keystr] = mirkey(baseName);
    keystr = get(keystr, 'Data');
    keystr = keystr{1};
    df(k).keyClarity = cell2mat(keystr);

    mode = mirmode(baseName);
    mode = get(mode, 'Data');
    mode = mode{1};
    df(k).majorness = cell2mat(mode);

    x = mirroughness(baseName);
    dat = get(x, 'Data');
    dat = dat{1};
    df(k).roughness = mean(dat{1});
    'completed Harmony'

    %%Register, 2
    pit = mirpitch(baseName);
    pit = get(pit, 'Data');
    pit = pit{1};
    df(k).pitch = cell2mat(pit{1});

    c = mirchromagram(baseName,'Frame','Wrap',0,'Pitch',0);
    cp = mirpeaks(c, 'Total',1);
    cp = get(cp,'PeakPosUnit');
    cp = cp{1};
    cp = cell2mat(cp{1});
    df(k).chromaStd = std(cp);
    'completed Register'

    %%Structure, 1
    x = mirnovelty(baseName);
    dat = get(x, 'Data');
    dat = dat{1};
    df(k).novelty = mean(dat{1});

    strcat('completed clip number ',string(k))
end

save('musicFeatures','df')
