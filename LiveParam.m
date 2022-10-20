classdef LiveParam < handle
    
    properties
        mountdir
        analysisdir
        datadir
        datafiles
        numfish
        
        n_randsamples = 1000;
        prcthresh = 0.2;
        prcthresh_param = 0.8;
        prcthresh_eyes = 0.06;
        areathresh_fish = 500;
        arearange_param = [10 120];
        arearange_eyes = [100 500];
        param_rad = 15;
        fishlen = 165;
        numSkelPoints = 6;
        
        tracking
        frame
        frame_num
        numframes
        gpu = 1; % 0: use CPU, 1: use GPU
    end
    
    methods
        
        %------------------------------------------------------------------
        function obj = LiveParam()
            % identify computer to use correct data path and number of cores 
            [ret, name] = system('hostname');
            computer = strtrim(lower(name));

            switch computer
                case 'thebeast'
                    maxNumCompThreads(6);
                    obj.mountdir = '/media/NFS2';
                case 'zfpc19'
                    maxNumCompThreads(68);
                    obj.mountdir = '/media/martin/NFS1';
                case 'bluebird'
                    maxNumCompThreads(8);
                    obj.mountdir = '/media/martin/DATA_1';
                otherwise
                    error('computer not recognized');
            end
            
            % path --------------------------------------------------------
            obj.analysisdir = fullfile(obj.mountdir,'MeCP2/Analysis');
            obj.datadir = fullfile(obj.mountdir,'MeCP2/LiveParamecia');
             
            scriptpath = genpath(obj.analysisdir);
            addpath(scriptpath);
            cd(obj.analysisdir);
            
            % files -------------------------------------------------------
            obj.loadfiles();
        end
        
        
        %------------------------------------------------------------------
        function loadfiles(obj)
            
            obj.datafiles.video = {
                fullfile(obj.datadir,'WT/2022_01_19_03.avi');
                fullfile(obj.datadir,'WT/2022_01_20_00.avi');
                fullfile(obj.datadir,'WT/2022_01_20_01.avi');
                fullfile(obj.datadir,'WT/2022_01_20_02.avi');
                fullfile(obj.datadir,'WT/2022_01_20_03.avi');
                fullfile(obj.datadir,'WT/2022_01_21_00.avi');
                fullfile(obj.datadir,'WT/2022_01_21_01.avi');
                fullfile(obj.datadir,'WT/2022_01_21_02.avi');
                fullfile(obj.datadir,'WT/2022_01_21_03.avi');
                fullfile(obj.datadir,'WT/2022_01_21_04.avi');
                fullfile(obj.datadir,'MeCP2/2022_01_26_00.avi');
                fullfile(obj.datadir,'MeCP2/2022_01_26_01.avi');
                fullfile(obj.datadir,'MeCP2/2022_01_26_02.avi');
                fullfile(obj.datadir,'MeCP2/2022_01_26_03.avi');
                fullfile(obj.datadir,'MeCP2/2022_01_26_04.avi');
                fullfile(obj.datadir,'MeCP2/2022_01_26_05.avi');
                fullfile(obj.datadir,'MeCP2/2022_01_26_06.avi');
                fullfile(obj.datadir,'MeCP2/2022_01_26_07.avi');
                };

            obj.datafiles.timestamp = {
                fullfile(obj.datadir,'WT/2022_01_19_03.txt');
                fullfile(obj.datadir,'WT/2022_01_20_00.txt');
                fullfile(obj.datadir,'WT/2022_01_20_01.txt');
                fullfile(obj.datadir,'WT/2022_01_20_02.txt');
                fullfile(obj.datadir,'WT/2022_01_20_03.txt');
                fullfile(obj.datadir,'WT/2022_01_21_00.txt');
                fullfile(obj.datadir,'WT/2022_01_21_01.txt');
                fullfile(obj.datadir,'WT/2022_01_21_02.txt');
                fullfile(obj.datadir,'WT/2022_01_21_03.txt');
                fullfile(obj.datadir,'WT/2022_01_21_04.txt');
                fullfile(obj.datadir,'MeCP2/2022_01_26_00.txt');
                fullfile(obj.datadir,'MeCP2/2022_01_26_01.txt');
                fullfile(obj.datadir,'MeCP2/2022_01_26_02.txt');
                fullfile(obj.datadir,'MeCP2/2022_01_26_03.txt');
                fullfile(obj.datadir,'MeCP2/2022_01_26_04.txt');
                fullfile(obj.datadir,'MeCP2/2022_01_26_05.txt');
                fullfile(obj.datadir,'MeCP2/2022_01_26_06.txt');
                fullfile(obj.datadir,'MeCP2/2022_01_26_07.txt');
                };
            
            assert(numel(obj.datafiles.video) == numel(obj.datafiles.timestamp),...
                'videos and timestamps not matching');
            obj.numfish = numel(obj.datafiles.video);
        end
        
        %------------------------------------------------------------------
        function background_model = model_background(obj,indfish,rangesamp)
            
            assert(indfish > 0 & indfish <= obj.numfish,'index out of range');
            assert(rangesamp(2)-rangesamp(1) >= obj.n_randsamples)

            disp(['computing background ' obj.datafiles.video{indfish} ':'])
            
            %frames_bckg = sort(randsample(rangesamp(1):rangesamp(2),obj.n_randsamples));
            frames_bckg = linspace(rangesamp(1),rangesamp(2),obj.n_randsamples);
            movie = VideoReader(obj.datafiles.video{indfish});
            background_model = zeros(movie.height,movie.width,obj.n_randsamples,'single');
            frame_num = 0;
            background_frame = 0;
            while movie.hasFrame()
                frame = movie.readFrame();
                frame_num = frame_num + 1;
                if ismember(frame_num,frames_bckg)
                     background_frame = background_frame + 1;
                     if mod(background_frame,obj.n_randsamples/20)==0
                        fprintf([num2str(round(100*(background_frame/obj.n_randsamples))) '%% '])
                     end
                     frame_b = im2single(frame);
                     background_model(:,:,background_frame) = squeeze(frame_b(:,:,1));
                end
            end
            fprintf('\n')
        end
        
        %------------------------------------------------------------------
        function detect_fish(obj,framefilt)
            
            obj.tracking.AdaptThreshFish(obj.frame_num) = prctile(framefilt(:),obj.prcthresh);
            bw = framefilt <= obj.tracking.AdaptThreshFish(obj.frame_num);
            if obj.gpu
                bw = bwareaopen(gather(bw),obj.areathresh_fish);
            else
                bw = bwareaopen(bw,obj.areathresh_fish);
            end
            [y,x] = find(bw);
            
            if ~isempty(x)
                [PC,scores] = pca([x y]);
                % make sure that PCs always point in the same direction
                if abs(max(scores(:,1))) < abs(min(scores(:,1)))
                    PC(:,1) = -PC(:,1);
                end
                if det(PC) < 0
                    PC(:,2) = -PC(:,2);
                end

                obj.tracking.PC1X(obj.frame_num) = PC(1,1);
                obj.tracking.PC1Y(obj.frame_num) = PC(2,1);
                obj.tracking.PC2X(obj.frame_num) = PC(1,2);
                obj.tracking.PC2Y(obj.frame_num) = PC(2,2);
                obj.tracking.CentroidX(obj.frame_num) = mean(x);
                obj.tracking.CentroidY(obj.frame_num) = mean(y);
                obj.tracking.Area(obj.frame_num) = length(x);
                
                % add fish principal axes to obj.frame
                linelength = 50;
                obj.frame = insertShape(obj.frame,'Line',...
                    [obj.tracking.CentroidX(obj.frame_num),...
                    obj.tracking.CentroidY(obj.frame_num),...
                    obj.tracking.CentroidX(obj.frame_num) - linelength*PC(1,1),...
                    obj.tracking.CentroidY(obj.frame_num) - linelength*PC(2,1);...
                    obj.tracking.CentroidX(obj.frame_num),...
                    obj.tracking.CentroidY(obj.frame_num),...
                    obj.tracking.CentroidX(obj.frame_num) + linelength*PC(1,2),...
                    obj.tracking.CentroidY(obj.frame_num) + linelength*PC(2,2)],...
                    'LineWidth',1,'Color',[0 0 0; 1 1 1]);
            end
        end
        
        %------------------------------------------------------------------
        function detect_paramecias(obj,framefilt)
            
            obj.tracking.AdaptThreshParam(obj.frame_num) = prctile(framefilt(:),obj.prcthresh_param);
            bw = framefilt <= obj.tracking.AdaptThreshParam(obj.frame_num);
            if obj.gpu
                bw = bwareafilt(gather(bw),obj.arearange_param);
            else
                bw = bwareafilt(bw,obj.arearange_param);
            end
            statParam = regionprops(bw,'Centroid');

            obj.tracking.numParam(obj.frame_num) = numel(statParam);

            % add circles around paramecias in the video
            if ~isempty(statParam)
                  param_coord = vertcat(statParam.Centroid);
                  obj.tracking.ParamCoords{obj.frame_num} = param_coord;
                  obj.frame = insertShape(obj.frame,'Circle',...
                  [param_coord obj.param_rad*ones(size(param_coord,1),1)],...
                    'LineWidth',1,'Color',[1 0 0]);
            end
        end
        
        %------------------------------------------------------------------
        function detect_eyes(obj,framefilt)
            
            if ~isnan(obj.tracking.PC1X(obj.frame_num))
                
                PC = [obj.tracking.PC1X(obj.frame_num) obj.tracking.PC2X(obj.frame_num);
                    obj.tracking.PC1Y(obj.frame_num) obj.tracking.PC2Y(obj.frame_num)];
                           
                obj.tracking.AdaptThreshEyes(obj.frame_num) = prctile(framefilt(:),obj.prcthresh_eyes);
                bw = framefilt <= obj.tracking.AdaptThreshEyes(obj.frame_num);
                if obj.gpu
                    bw = bwareafilt(gather(bw),obj.arearange_eyes);
                else
                    bw = bwareafilt(bw,obj.arearange_eyes);
                end
                statEye = regionprops(bw,...
                    'Area',...
                    'Orientation',...
                    'Centroid',...
                    'PixelIdxList',...
                    'MajorAxisLength',...
                    'MinorAxisLength');
                if ~isempty(statEye)
                    % project coordinates in PC space
                    eyes_coord = vertcat(statEye.Centroid);
                    eyes_coord_PC = (eyes_coord - [obj.tracking.CentroidX(obj.frame_num) obj.tracking.CentroidY(obj.frame_num)]) * PC;
                    left_eye = find(eyes_coord_PC(:,1) < -10 & eyes_coord_PC(:,2) < -5);
                    right_eye = find(eyes_coord_PC(:,1) < -10 & eyes_coord_PC(:,2) > 5);
                    angle = atan2(obj.tracking.PC1Y(obj.frame_num),obj.tracking.PC1X(obj.frame_num));
                    Ctd = [obj.tracking.CentroidX(obj.frame_num) obj.tracking.CentroidY(obj.frame_num)];
                    
                    % color the eyes in blue and red in the video
                    if length(left_eye) == 1
                        coordEyeLeftMajor = statEye(left_eye).MajorAxisLength./2 * [1; -1] ...
                            * [cosd(-statEye(left_eye).Orientation) sind(-statEye(left_eye).Orientation)] ...
                            + statEye(left_eye).Centroid;
                         coordEyeLeftMajor_PC = (coordEyeLeftMajor - Ctd) * PC;

                         obj.frame = insertShape(obj.frame,'Line',...
                            [coordEyeLeftMajor(1,1),...
                            coordEyeLeftMajor(1,2),...
                            coordEyeLeftMajor(2,1),...
                            coordEyeLeftMajor(2,2)],...
                            'LineWidth',1,'Color',[1 0 0]);
                        
                        vect_PC = coordEyeLeftMajor_PC(2,:) - coordEyeLeftMajor_PC(1,:);
                        if vect_PC(1) < 0
                            vect_PC = -vect_PC;
                            obj.frame = insertShape(obj.frame,'FilledCircle',...
                            [coordEyeLeftMajor(2,1),...
                            coordEyeLeftMajor(2,2),...
                            2],...
                            'LineWidth',1,'Color',[1 0 0]);
                        else
                            obj.frame = insertShape(obj.frame,'FilledCircle',...
                            [coordEyeLeftMajor(1,1),...
                            coordEyeLeftMajor(1,2),...
                            2],...
                            'LineWidth',1,'Color',[1 0 0]);
                        end
                        
                        obj.tracking.LeftEyeAngle(obj.frame_num) = atan2(vect_PC(2),vect_PC(1));

                    end

                    if length(right_eye) == 1
                        coordEyeRightMajor = statEye(right_eye).MajorAxisLength./2 * [1; -1] ...
                            * [cosd(-statEye(right_eye).Orientation) sind(-statEye(right_eye).Orientation)] ...
                            + statEye(right_eye).Centroid;
                        coordEyeRightMajor_PC = (coordEyeRightMajor - Ctd) * PC;

                        obj.frame = insertShape(obj.frame,'Line',...
                            [coordEyeRightMajor(1,1),...
                            coordEyeRightMajor(1,2),...
                            coordEyeRightMajor(2,1),...
                            coordEyeRightMajor(2,2)],...
                            'LineWidth',1,'Color',[0 1 1]);
                        
                        vect_PC = coordEyeRightMajor_PC(2,:) - coordEyeRightMajor_PC(1,:);
                        if  vect_PC(1) < 0
                            vect_PC = -vect_PC;
                            obj.frame = insertShape(obj.frame,'FilledCircle',...
                            [coordEyeRightMajor(2,1),...
                            coordEyeRightMajor(2,2),...
                            2],...
                            'LineWidth',1,'Color',[0 1 1]);
                        else
                            obj.frame = insertShape(obj.frame,'FilledCircle',...
                            [coordEyeRightMajor(1,1),...
                            coordEyeRightMajor(1,2),...
                            2],...
                            'LineWidth',1,'Color',[0 1 1]);
                        end
                        
                        obj.tracking.RightEyeAngle(obj.frame_num) = atan2(vect_PC(2),vect_PC(1));
                    end
                end
            end
        end
        
        %------------------------------------------------------------------
        function detect_tail(obj,framefilt)
            
            skel_radius = obj.fishlen / (obj.numSkelPoints-1);
            theta =  -pi/2:asin(1/skel_radius):pi/2;
            radius = [1:1:skel_radius]';
             
            pad_value = obj.fishlen;
            framefilt_padded = padarray(framefilt,[pad_value pad_value],NaN);
            
            bw = framefilt_padded <= obj.tracking.AdaptThreshFish(obj.frame_num);
            if obj.gpu
                bw = bwareaopen(gather(bw),obj.areathresh_fish);
            else
                bw = bwareaopen(bw,obj.areathresh_fish);
            end
            [y,x] = find(bw);
            
            if ~isempty(x)
                x_0 = round(mean(x));
                y_0 = round(mean(y));
                obj.tracking.Skeleton(obj.frame_num,:,1) = [x_0 y_0] - pad_value;
                angle = atan2(obj.tracking.PC1Y(obj.frame_num),obj.tracking.PC1X(obj.frame_num));
                best_theta = angle;
                for s = 2:obj.numSkelPoints
                    x_grid = round(radius*cos(theta+best_theta));
                    y_grid = round(radius*sin(theta+best_theta));
                    x_ = x_0 + x_grid(:); 
                    y_ = y_0 + y_grid(:);
                    index = sub2ind(size(framefilt_padded),y_,x_);
                    values = framefilt_padded(index);
                    values = reshape(values,length(radius),length(theta));
                    [~,min_pos] = min(sum(values));
                    best_theta = theta(min_pos)+best_theta;
                    x_0 = x_0 + round(skel_radius*cos(best_theta));
                    y_0 = y_0 + round(skel_radius*sin(best_theta));
                    obj.tracking.Skeleton(obj.frame_num,1,s) = x_0 - pad_value;
                    obj.tracking.Skeleton(obj.frame_num,2,s) = y_0 - pad_value;
                    obj.tracking.SkeletonAngle(obj.frame_num,s) = best_theta - angle;
                    
                    obj.frame = insertShape(obj.frame,'Line',...
                        [obj.tracking.Skeleton(obj.frame_num,1,s-1),...
                        obj.tracking.Skeleton(obj.frame_num,2,s-1),...
                        obj.tracking.Skeleton(obj.frame_num,1,s),...
                        obj.tracking.Skeleton(obj.frame_num,2,s)],...
                        'LineWidth',1,'Color',[0 1 0]);
                end

                %%TODO fit smoothing splines to the tail (todo in PC space)
%                 p = 0.9;
%                 skel_x = obj.tracking.Skeleton(obj.frame_num,1,:);
%                 skel_y = obj.tracking.Skeleton(obj.frame_num,2,:);
%                 x_up = skel_x(1):4:skel_x(end);
%                 [~,y_up,~] = csaps(skel_x,skel_y,p,x_up);
%                 for s = 1:length(x_up)
%                     obj.frame = insertShape(obj.frame,'FilledCircle',...
%                         [x_up(s) y_up(s) 1.5],'LineWidth',1,'Color',[0 1 0]);
%                 end
            end
        end
        
        %------------------------------------------------------------------
        function frame_zoomed = zoom_and_rotate(obj)
        %%TODO fix magic numbers as number of fish length
            
            if ~isnan(obj.tracking.PC1X(obj.frame_num))
                angle = atan2(obj.tracking.PC1Y(obj.frame_num),obj.tracking.PC1X(obj.frame_num));

                T0 = [1 0 0;
                      0 1 0;
                      -obj.tracking.CentroidX(obj.frame_num) -obj.tracking.CentroidY(obj.frame_num) 1];

                R = [cos(angle)  -sin(angle) 0;
                     sin(angle) cos(angle) 0;
                     0 0 1];

                T1 = [1 0 0;
                     0 1 0;
                     size(obj.frame,2)/2 size(obj.frame,1)/2 1];

                Tf = T0 * R * T1;

                tform = affine2d(Tf);  
                RA = imref2d(size(obj.frame));
                frame_zoomed = imwarp(obj.frame,tform,'OutputView',RA);
                frame_zoomed = imcrop(frame_zoomed,...
                    [size(obj.frame,2)/2-150 size(obj.frame,1)/2-100 350 200]);
            else
                frame_zoomed = zeros(201,351);
            end
            
        end
        
        %------------------------------------------------------------------
        function track(obj,indfish)
            
            movie = VideoReader(obj.datafiles.video{indfish});
            [path,prefix,~] = fileparts(obj.datafiles.video{indfish});
            imagedir = fullfile(path,prefix);
            
            disp(['geting number of frames ' obj.datafiles.video{indfish} ':'])
            obj.numframes = movie.NumFrames;
            
            if obj.gpu
                background_model = gpuArray(obj.model_background(indfish,[1 obj.numframes]));
            else
                background_model = obj.model_background(indfish,[1 obj.numframes]);
            end
            
            disp(['analyzing video ' obj.datafiles.video{indfish} ':'])
    
            obj.tracking.PC1X = NaN(obj.numframes,1);
            obj.tracking.PC1Y = NaN(obj.numframes,1);
            obj.tracking.PC2X = NaN(obj.numframes,1);
            obj.tracking.PC2Y = NaN(obj.numframes,1);
            obj.tracking.CentroidX = NaN(obj.numframes,1);
            obj.tracking.CentroidY = NaN(obj.numframes,1);
            obj.tracking.Area = NaN(obj.numframes,1);
            obj.tracking.numParam = NaN(obj.numframes,1);
            obj.tracking.AdaptThreshFish = NaN(obj.numframes,1);
            obj.tracking.AdaptThreshParam = NaN(obj.numframes,1);
            obj.tracking.AdaptThreshEyes = NaN(obj.numframes,1);
            obj.tracking.ParamCoords = cell(obj.numframes,1);
            obj.tracking.LeftEyeAngle = NaN(obj.numframes,1);
            obj.tracking.RightEyeAngle = NaN(obj.numframes,1);
            obj.tracking.Skeleton = NaN(obj.numframes,2,obj.numSkelPoints);
            obj.tracking.SkeletonAngle = NaN(obj.numframes,obj.numSkelPoints);
            
            vOut1 = VideoWriter([imagedir '_tracking_offline.avi'],'Motion JPEG AVI');
            vOut1.Quality = 80;
            vOut1.FrameRate = 100;
            open(vOut1);

            vOut2 = VideoWriter([imagedir '_cropped.avi'],'Motion JPEG AVI');
            vOut2.Quality = 80;
            vOut2.FrameRate = 100;
            open(vOut2);
            
            obj.frame_num = 0;
            tic
            while movie.hasFrame()
                obj.frame = movie.readFrame();
                obj.frame_num = obj.frame_num + 1;
                if obj.gpu
                    obj.frame = gpuArray(im2single(squeeze(obj.frame(:,:,1))));
                else
                    obj.frame = im2single(squeeze(obj.frame(:,:,1)));
                end
                
                if mod(obj.frame_num,round(obj.numframes * 1/100))==0
                    fprintf([num2str(round(100 * obj.frame_num/obj.numframes)) '%% '])
                    toc
                end

                framesub = obj.frame-background_model;
                framefilt = medfilt2(sum(framesub,3),[3 3]);
                if obj.gpu
                    obj.frame = gather(obj.frame);
                    clear framesub;
                end
                
                obj.detect_fish(framefilt);
                obj.detect_paramecias(framefilt);
                obj.detect_eyes(framefilt);
                obj.detect_tail(framefilt);

                frame_zoomed = obj.zoom_and_rotate;
                
                writeVideo(vOut1,obj.frame)
                writeVideo(vOut2,frame_zoomed)
            end
            
            close(vOut1);
            close(vOut2);
            
            behavior = obj.tracking;
            save([imagedir '_tracking_offline.mat'],'behavior');
        end
        
        %------------------------------------------------------------------
        function analyse(obj)
            
            for i = 1:obj.numfish
                obj.track(i)
            end
        end
    end
end
