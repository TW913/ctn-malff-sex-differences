% Batch-normalize resting-state BOLD images to MNI space through each
% subject's T1w image with SPM12.
%
% Input data are expected to be BIDS-like folders under INPUT_DIR, for
% example:
%   D:\openneuro data\sub-009\anat\sub-009_run-01_T1w.nii.gz
%   D:\openneuro data\sub-009\func\sub-009_task-rest_bold.nii.gz

clear;
clc;

SPM_PATH = 'M:\Documents\MATLAB\spm12';

INPUT_DIR = 'D:\openneuro data';
BASE_DIR = fileparts(mfilename('fullpath'));

TPM_FILE = fullfile(SPM_PATH, 'tpm', 'TPM.nii');
OUT_DIR = 'D:\output';
WORK_ROOT = fullfile(OUT_DIR, 'spm_work_batch_to_mni');
STATUS_CSV = fullfile(WORK_ROOT, 'batch_status.csv');

SUBJECT_PATTERN = 'sub-*';
BOLD_PATTERN = '*task-rest*bold.nii*';
T1W_PATTERN = '*T1w.nii*';

% 3 mm keeps 4D fMRI outputs tractable. Set to [1 1 1] only if you really
% need 1 mm BOLD outputs and have enough disk/memory for the full batch.
OUTPUT_VOXEL_SIZE = [2 2 2];

% Match the single-subject M script output field-of-view.
NORMALISE_BB = [-78 -112 -70; 78 76 85];

% Leave false to make long batch runs resumable. Existing valid outputs will
% be skipped, so rerunning the script continues with unfinished BOLD files.
OVERWRITE_OUTPUT = false;
OVERWRITE_PREPARED_NIFTI = false;
CONTINUE_ON_ERROR = true;

addpath(SPM_PATH);

if exist('spm', 'file') ~= 2
    error('SPM was not found. Check SPM_PATH: %s', SPM_PATH);
end

ensure_file(TPM_FILE, 'SPM tissue probability map');
ensure_existing_dir(INPUT_DIR, 'input dataset');
ensure_dir(OUT_DIR);
ensure_dir(WORK_ROOT);

spm('Defaults', 'fMRI');
spm_jobman('initcfg');

normalise_bb = NORMALISE_BB;
subjects = find_subject_dirs(INPUT_DIR, SUBJECT_PATTERN);

if isempty(subjects)
    error('No subject folders matching %s were found in: %s', ...
        SUBJECT_PATTERN, INPUT_DIR);
end

fprintf('Input dataset: %s\n', INPUT_DIR);
fprintf('Normalise bounding box: [%g %g %g; %g %g %g]\n', ...
    normalise_bb(1, 1), normalise_bb(1, 2), normalise_bb(1, 3), ...
    normalise_bb(2, 1), normalise_bb(2, 2), normalise_bb(2, 3));
fprintf('Output voxel size: [%g %g %g]\n', OUTPUT_VOXEL_SIZE);
fprintf('Output directory: %s\n', OUT_DIR);
fprintf('Working directory: %s\n', WORK_ROOT);
fprintf('Subject folders found: %d\n\n', numel(subjects));

status_rows = cell(0, 6);

for s = 1:numel(subjects)
    subject_dir = subjects{s};
    [~, subject_id] = fileparts(subject_dir);

    fprintf('\n============================================================\n');
    fprintf('Subject %d/%d: %s\n', s, numel(subjects), subject_id);
    fprintf('============================================================\n');

    try
        t1w_source = select_t1w_file(subject_dir, T1W_PATTERN);
        bold_sources = select_bold_files(subject_dir, BOLD_PATTERN);

        if isempty(t1w_source)
            message = sprintf('No T1w NIfTI matching %s', T1W_PATTERN);
            fprintf('SKIP: %s\n', message);
            status_rows = append_status(status_rows, subject_id, '', '', ...
                'SKIPPED', message);
            write_status_csv(STATUS_CSV, status_rows);
            continue;
        end

        if isempty(bold_sources)
            message = sprintf('No BOLD NIfTI matching %s', BOLD_PATTERN);
            fprintf('SKIP: %s\n', message);
            status_rows = append_status(status_rows, subject_id, '', '', ...
                'SKIPPED', message);
            write_status_csv(STATUS_CSV, status_rows);
            continue;
        end

        fprintf('Selected T1w: %s\n', t1w_source);
        fprintf('BOLD runs found: %d\n', numel(bold_sources));

        for b = 1:numel(bold_sources)
            bold_source = bold_sources{b};
            output_file = make_output_file(OUT_DIR, bold_source);

            fprintf('\nSubject %s, BOLD %d/%d\n', ...
                subject_id, b, numel(bold_sources));
            fprintf('Input BOLD: %s\n', bold_source);
            fprintf('Output BOLD: %s\n', output_file);

            if should_skip_existing_output(output_file, OVERWRITE_OUTPUT)
                message = 'Valid output already exists; set OVERWRITE_OUTPUT=true to rerun';
                fprintf('SKIP: %s\n', message);
                status_rows = append_status(status_rows, subject_id, ...
                    bold_source, output_file, 'SKIPPED', message);
                write_status_csv(STATUS_CSV, status_rows);
                continue;
            end

            try
                run_bold_to_mni( ...
                    subject_id, bold_source, t1w_source, output_file, ...
                    WORK_ROOT, TPM_FILE, normalise_bb, OUTPUT_VOXEL_SIZE, ...
                    OVERWRITE_OUTPUT, OVERWRITE_PREPARED_NIFTI);

                message = 'Done';
                status_rows = append_status(status_rows, subject_id, ...
                    bold_source, output_file, 'DONE', message);
                write_status_csv(STATUS_CSV, status_rows);
            catch ME
                message = clean_message(ME.message);
                fprintf(2, 'FAILED: %s\n', message);
                status_rows = append_status(status_rows, subject_id, ...
                    bold_source, output_file, 'FAILED', message);
                write_status_csv(STATUS_CSV, status_rows);

                if ~CONTINUE_ON_ERROR
                    rethrow(ME);
                end
            end
        end
    catch ME
        message = clean_message(ME.message);
        fprintf(2, 'FAILED subject setup: %s\n', message);
        status_rows = append_status(status_rows, subject_id, '', '', ...
            'FAILED', message);
        write_status_csv(STATUS_CSV, status_rows);

        if ~CONTINUE_ON_ERROR
            rethrow(ME);
        end
    end
end

fprintf('\nBatch finished.\n');
fprintf('Status CSV: %s\n', STATUS_CSV);
fprintf('MNI outputs: %s\n', OUT_DIR);


function run_bold_to_mni(subject_id, bold_source, t1w_source, output_file, ...
    work_root, tpm_file, normalise_bb, output_voxel_size, overwrite_output, ...
    overwrite_prepared_nifti)

    bold_stem = nifti_stem(bold_source);
    run_dir = fullfile(work_root, subject_id, safe_filename(bold_stem));
    nifti_dir = fullfile(run_dir, 'nifti');
    ensure_dir(run_dir);
    ensure_dir(nifti_dir);

    bold_file = prepare_nifti(bold_source, nifti_dir, overwrite_prepared_nifti);
    t1w_file = prepare_nifti(t1w_source, nifti_dir, overwrite_prepared_nifti);

    [t1w_folder, t1w_name, t1w_ext] = fileparts(t1w_file);
    deformation_file = fullfile(t1w_folder, ['y_' t1w_name t1w_ext]);
    warped_tissue_files = { ...
        fullfile(t1w_folder, ['wc1' t1w_name t1w_ext]); ...
        fullfile(t1w_folder, ['wc2' t1w_name t1w_ext]); ...
        fullfile(t1w_folder, ['wc3' t1w_name t1w_ext])};

    run_id = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    split_dir = fullfile(run_dir, ['split_' run_id]);
    ensure_dir(split_dir);

    if exist(output_file, 'file') == 2 && overwrite_output
        delete(output_file);
    end

    fprintf('Prepared BOLD: %s\n', bold_file);
    fprintf('Prepared T1w: %s\n', t1w_file);
    fprintf('Split directory: %s\n', split_dir);

    fprintf('Step 1/6: splitting 4D BOLD into 3D volumes...\n');
    bold_headers = spm_vol(bold_file);
    if numel(bold_headers) < 2
        error('Expected a 4D BOLD image with multiple volumes: %s', bold_file);
    end
    split_headers = spm_file_split(bold_headers, split_dir);
    bold_vols = headers_to_vol_list(split_headers);

    if isempty(bold_vols)
        error('No split BOLD volumes were created in: %s', split_dir);
    end

    fprintf('Number of BOLD volumes: %d\n', numel(bold_vols));

    matlabbatch = [];
    matlabbatch{1}.spm.spatial.realign.estwrite.data = {bold_vols};
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.sep = 4;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.rtm = 1;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.interp = 2;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.weight = '';
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.which = [2 1];
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.interp = 4;
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.mask = 1;
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.prefix = 'r';

    fprintf('Step 2/6: realigning BOLD volumes...\n');
    spm_jobman('run', matlabbatch);

    [split_folder, first_name, first_ext] = fileparts(get_header_fname(split_headers, 1));
    mean_bold = fullfile(split_folder, ['mean' first_name first_ext]);
    ensure_file(mean_bold, 'mean BOLD created by SPM');

    realigned_bold_vols = prefixed_vol_list(split_headers, 'r');
    ensure_volume_files(realigned_bold_vols, 'realigned BOLD volume');

    matlabbatch = [];
    matlabbatch{1}.spm.spatial.coreg.estimate.ref = {[t1w_file ',1']};
    matlabbatch{1}.spm.spatial.coreg.estimate.source = {[mean_bold ',1']};
    matlabbatch{1}.spm.spatial.coreg.estimate.other = realigned_bold_vols;
    matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
    matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
    matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.tol = ...
        [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];

    fprintf('Step 3/6: coregistering mean/realigned BOLD to T1w...\n');
    spm_jobman('run', matlabbatch);

    matlabbatch = [];
    matlabbatch{1}.spm.spatial.preproc.channel.vols = {[t1w_file ',1']};
    matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
    matlabbatch{1}.spm.spatial.preproc.channel.write = [0 1];
    matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {[tpm_file ',1']};
    matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1;
    matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [1 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {[tpm_file ',2']};
    matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 1;
    matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [1 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {[tpm_file ',3']};
    matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
    matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [1 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {[tpm_file ',4']};
    matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 3;
    matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm = {[tpm_file ',5']};
    matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus = 4;
    matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm = {[tpm_file ',6']};
    matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
    matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
    matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
    matlabbatch{1}.spm.spatial.preproc.warp.write = [0 1];

    fprintf('Step 4/6: segmenting T1w and estimating T1w-to-MNI deformation...\n');
    spm_jobman('run', matlabbatch);
    ensure_file(deformation_file, 'forward deformation field created by SPM segment');
    fprintf('MNI-space tissue maps:\n');
    report_optional_file(warped_tissue_files{1}, 'GM probability map');
    report_optional_file(warped_tissue_files{2}, 'WM probability map');
    report_optional_file(warped_tissue_files{3}, 'CSF probability map');

    matlabbatch = [];
    matlabbatch{1}.spm.spatial.normalise.write.subj.def = {deformation_file};
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = realigned_bold_vols;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = normalise_bb;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = output_voxel_size;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'w';

    fprintf('Step 5/6: applying T1w deformation field to BOLD volumes...\n');
    spm_jobman('run', matlabbatch);

    normalized_bold_vols = prefixed_vol_list(split_headers, 'wr');
    ensure_volume_files(normalized_bold_vols, 'normalized BOLD volume');

    fprintf('Step 6/6: merging normalized 3D volumes into one 4D file...\n');
    spm_file_merge(normalized_bold_vols, output_file, 16);
    ensure_file(output_file, 'merged normalized 4D BOLD');

    fprintf('Done: %s\n', output_file);
end


function ensure_file(file_path, label)
    if exist(file_path, 'file') ~= 2
        error('%s file not found: %s', label, file_path);
    end
end


function ensure_dir(dir_path)
    if exist(dir_path, 'dir') ~= 7
        mkdir(dir_path);
    end
end


function ensure_existing_dir(dir_path, label)
    if exist(dir_path, 'dir') ~= 7
        error('%s directory not found: %s', label, dir_path);
    end
end


function subjects = find_subject_dirs(input_dir, subject_pattern)
    ensure_existing_dir(input_dir, 'input dataset');
    entries = dir(fullfile(input_dir, subject_pattern));
    entries = entries([entries.isdir]);
    names = cell(numel(entries), 1);
    for i = 1:numel(entries)
        names{i} = lower(entries(i).name);
    end
    [~, order] = sort(names);
    entries = entries(order);

    subjects = cell(numel(entries), 1);
    for i = 1:numel(entries)
        subjects{i} = fullfile(input_dir, entries(i).name);
    end
end


function t1w_file = select_t1w_file(subject_dir, t1w_pattern)
    files = find_files_recursive(fullfile(subject_dir, 'anat'), t1w_pattern);
    if isempty(files)
        files = find_files_recursive(subject_dir, t1w_pattern);
        files = filter_path_segment(files, 'anat');
    end
    files = filter_nifti_files(files);
    files = sort_file_paths(files);
    files = prefer_uncompressed_nifti(files);

    if isempty(files)
        t1w_file = '';
        return;
    end

    run01 = {};
    for i = 1:numel(files)
        if contains(lower(files{i}), 'run-01')
            run01{end + 1, 1} = files{i}; %#ok<AGROW>
        end
    end

    if ~isempty(run01)
        t1w_file = run01{1};
    else
        t1w_file = files{1};
    end
end


function bold_files = select_bold_files(subject_dir, bold_pattern)
    bold_files = find_files_recursive(fullfile(subject_dir, 'func'), bold_pattern);
    if isempty(bold_files)
        bold_files = find_files_recursive(subject_dir, bold_pattern);
        bold_files = filter_path_segment(bold_files, 'func');
    end
    bold_files = filter_nifti_files(bold_files);
    bold_files = sort_file_paths(bold_files);
    bold_files = prefer_uncompressed_nifti(bold_files);
end


function files = find_files_recursive(root_dir, pattern)
    files = {};
    if exist(root_dir, 'dir') ~= 7
        return;
    end

    matches = dir(fullfile(root_dir, pattern));
    for i = 1:numel(matches)
        if ~matches(i).isdir
            files{end + 1, 1} = fullfile(root_dir, matches(i).name); %#ok<AGROW>
        end
    end

    entries = dir(root_dir);
    for i = 1:numel(entries)
        if entries(i).isdir && ...
                ~strcmp(entries(i).name, '.') && ...
                ~strcmp(entries(i).name, '..')
            child_dir = fullfile(root_dir, entries(i).name);
            child_files = find_files_recursive(child_dir, pattern);
            files = [files; child_files]; %#ok<AGROW>
        end
    end
end


function files = filter_nifti_files(files)
    keep = false(numel(files), 1);
    for i = 1:numel(files)
        lower_file = lower(files{i});
        keep(i) = ends_with(lower_file, '.nii') || ends_with(lower_file, '.nii.gz');
    end
    files = files(keep);
end


function files = filter_path_segment(files, segment)
    marker = lower([filesep segment filesep]);
    keep = false(numel(files), 1);
    for i = 1:numel(files)
        keep(i) = contains(lower(files{i}), marker);
    end
    files = files(keep);
end


function files = sort_file_paths(files)
    keys = cell(numel(files), 1);
    for i = 1:numel(files)
        keys{i} = lower(files{i});
    end
    [~, order] = sort(keys);
    files = files(order);
end


function files = prefer_uncompressed_nifti(files)
    selected = containers.Map('KeyType', 'char', 'ValueType', 'char');
    order = {};

    for i = 1:numel(files)
        key = lower(nifti_stem(files{i}));
        if ~isKey(selected, key)
            selected(key) = files{i};
            order{end + 1, 1} = key; %#ok<AGROW>
        elseif is_gz_nifti(selected(key)) && ~is_gz_nifti(files{i})
            selected(key) = files{i};
        end
    end

    files = cell(numel(order), 1);
    for i = 1:numel(order)
        files{i} = selected(order{i});
    end
end


function nifti_file = prepare_nifti(source_file, dest_dir, overwrite_existing)
    ensure_dir(dest_dir);
    source_stem = nifti_stem(source_file);
    nifti_file = fullfile(dest_dir, [source_stem '.nii']);

    if exist(nifti_file, 'file') == 2
        if overwrite_existing
            delete(nifti_file);
        else
            try
                validate_nifti_file(nifti_file);
                return;
            catch ME
                warning('Existing prepared NIfTI is invalid and will be recreated: %s\nReason: %s', ...
                    nifti_file, clean_message(ME.message));
                delete(nifti_file);
            end
        end
    end

    try
        if ends_with(lower(source_file), '.nii.gz')
            gunzip(source_file, dest_dir);
        elseif ends_with(lower(source_file), '.nii')
            copyfile(source_file, nifti_file);
        else
            error('Unsupported NIfTI path: %s', source_file);
        end
    catch ME
        if exist(nifti_file, 'file') == 2
            delete(nifti_file);
        end
        error('Failed to prepare NIfTI from source: %s. %s', ...
            source_file, clean_message(ME.message));
    end

    ensure_file(nifti_file, 'prepared NIfTI');
    validate_nifti_file(nifti_file);
end


function validate_nifti_file(nifti_file)
    try
        headers = spm_vol(nifti_file);
        if isempty(headers)
            error('SPM could not read any volumes.');
        end
        spm_read_vols(headers(1));
        if numel(headers) > 1
            spm_read_vols(headers(end));
        end
    catch ME
        error('Invalid NIfTI file: %s. %s', nifti_file, clean_message(ME.message));
    end
end


function output_file = make_output_file(out_dir, bold_source)
    ensure_dir(out_dir);
    bold_stem = nifti_stem(bold_source);
    output_file = fullfile(out_dir, [bold_stem '_space-MNI152_spm.nii']);
end


function tf = should_skip_existing_output(output_file, overwrite_output)
    tf = false;

    if overwrite_output || exist(output_file, 'file') ~= 2
        return;
    end

    try
        validate_nifti_file(output_file);
        tf = true;
    catch ME
        warning('Existing output is invalid and will be rerun: %s\nReason: %s', ...
            output_file, clean_message(ME.message));
        delete(output_file);
    end
end


function stem = nifti_stem(file_path)
    [~, name, ext] = fileparts(file_path);
    if strcmpi(ext, '.gz')
        [~, name2, ext2] = fileparts(name);
        if strcmpi(ext2, '.nii')
            stem = name2;
        else
            stem = name;
        end
    else
        stem = name;
    end
end


function safe_name = safe_filename(name)
    safe_name = regexprep(name, '[^A-Za-z0-9_-]', '_');
end


function vols = headers_to_vol_list(headers)
    vols = cell(numel(headers), 1);
    for i = 1:numel(headers)
        vols{i} = [get_header_fname(headers, i) ',1'];
    end
end


function vols = prefixed_vol_list(headers, prefix)
    vols = cell(numel(headers), 1);
    for i = 1:numel(headers)
        [folder, name, ext] = fileparts(get_header_fname(headers, i));
        vols{i} = [fullfile(folder, [prefix name ext]) ',1'];
    end
end


function fname = get_header_fname(headers, index)
    if isstruct(headers)
        fname = headers(index).fname;
    elseif iscell(headers)
        item = headers{index};
        if isstruct(item)
            fname = item.fname;
        else
            fname = item;
        end
    else
        fname = headers(index).fname;
    end
end


function ensure_volume_files(vols, label)
    for i = 1:numel(vols)
        file_path = regexprep(vols{i}, ',\d+$', '');
        ensure_file(file_path, label);
    end
end


function report_optional_file(file_path, label)
    if exist(file_path, 'file') == 2
        fprintf('  %s\n', file_path);
    else
        warning('%s was not found, continuing because it is not required for BOLD normalization: %s', ...
            label, file_path);
    end
end


function rows = append_status(rows, subject_id, bold_file, output_file, status, message)
    rows(end + 1, :) = { ...
        char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), ...
        subject_id, bold_file, output_file, status, clean_message(message)};
end


function write_status_csv(csv_file, rows)
    [csv_dir, ~, ~] = fileparts(csv_file);
    ensure_dir(csv_dir);

    fid = fopen(csv_file, 'w');
    if fid < 0
        error('Could not open status CSV for writing: %s', csv_file);
    end

    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, '"Timestamp","Subject","BoldFile","OutputFile","Status","Message"\n');
    for i = 1:size(rows, 1)
        fprintf(fid, '%s,%s,%s,%s,%s,%s\n', ...
            csv_escape(rows{i, 1}), csv_escape(rows{i, 2}), ...
            csv_escape(rows{i, 3}), csv_escape(rows{i, 4}), ...
            csv_escape(rows{i, 5}), csv_escape(rows{i, 6}));
    end
end


function text = csv_escape(value)
    if isempty(value)
        value = '';
    elseif isnumeric(value)
        value = num2str(value);
    elseif islogical(value)
        value = mat2str(value);
    end

    text = char(value);
    text = clean_message(text);
    text = strrep(text, '"', '""');
    text = ['"' text '"'];
end


function message = clean_message(message)
    message = char(message);
    message = regexprep(message, '\r|\n', ' ');
end


function tf = ends_with(text, suffix)
    tf = numel(text) >= numel(suffix) && ...
        strcmp(text(end - numel(suffix) + 1:end), suffix);
end


function tf = is_gz_nifti(file_path)
    tf = ends_with(lower(file_path), '.nii.gz');
end
