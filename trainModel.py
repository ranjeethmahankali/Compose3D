# from model_0 import *
from model import *
import shutil

rhinoDataset = dataset('dataset/')
# ballDataset = dataset('ball_dataset/')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # deleting old logs and setting up new ones
    shutil.rmtree(log_dir, ignore_errors=True)
    train_writer, test_writer = getSummaryWriters(sess)
    # loadModel(sess, model_save_path[0])
    loadModel(sess, model_save_path[1])

    # cycles = 6000
    cycles = 100000
    testStep = 20
    saveStep = 1500
    log_step = 5
    startTime = time.time()
    try:
        for i in range(cycles):
            batch = rhinoDataset.next_batch(batch_size)
            _, summary = sess.run([optim, merged], feed_dict={
                view_placeholder: batch[0],
                scene_params_placeholder: batch[1],
                keep_prob_placeholder: 0.7
            })
            if i % log_step == 0: train_writer.add_summary(summary,i)

            timer = estimate_time(startTime, cycles, i)
            pL = 10 # this is the length of the progress bar to be displayed
            pNum = i % pL
            pBar = '#'*pNum + ' '*(pL - pNum)

            # sys.stdout.write('...Training...|%s|-(%s/%s)- %s\r'%(pBar, i, cycles, timer))

            if i % testStep == 0:
                testBatch = rhinoDataset.test_batch(batch_size)
                lossVal, acc, v, outvals = sess.run([loss, accTensor, scene_params, output], feed_dict={
                    view_placeholder: testBatch[0],
                    scene_params_placeholder: testBatch[1],
                    keep_prob_placeholder: 1
                })
                test_writer.add_summary(summary, i)
                # print(outvals)
                # print(v)
                # print(testBatch[1])
                print('Accuracy: %.2f; Loss: %.2f; Sums: %.2f, %.2f%s'%(acc, lossVal, 
                        v.sum(),testBatch[1].sum(),' '*50))
        
        # now saving the trained model every 1500 cycles
            if i % saveStep == 0 and i != 0:
                saveModel(sess, model_save_path[1])
        
        # saving the model in the end
        saveModel(sess, model_save_path[1])
    # if the training is interrupted from keyboard (ctrl + c)
    except KeyboardInterrupt:
        print('')
        print('You interrupted the training process')
        decision = input('Do you want to save the current model before exiting? (y/n):')

        if decision == 'y':
            saveModel(sess, model_save_path[1])
        elif decision == 'n':
            print('\n...Model not saved...')
            pass
        else:
            print('\n...Invalid input. Model not saved...')
            pass